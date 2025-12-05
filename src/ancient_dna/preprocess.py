"""
preprocess.py
-----------------
数据预处理（Preprocessing）模块
Preprocessing module for genotype matrix cleaning, filtering, and imputation.

用于执行基因型矩阵的缺失值检测、样本与注释表对齐、缺失率计算、过滤与填补。
Used for performing missing-value analysis, alignment between genotype and metadata tables,
and imputation using multiple optimized algorithms (mode, mean, KNN, BallTree, Faiss, etc.).

功能：
    - align_by_id(): 对齐基因型矩阵与样本注释表；
    - compute_missing_rates(): 计算样本与SNP缺失率；
    - filter_by_missing(): 按缺失率阈值过滤；
    - impute_missing(): 通用缺失值填补入口；
    - grouped_imputation(): 按标签分组填补；
    - _fill_mode(), _fill_mean(), _fill_knn() 等: 各种填补方法实现。

Functions:
    - align_by_id(): Align genotype matrix with metadata table by sample IDs.
    - compute_missing_rates(): Compute missing rate per sample and per SNP.
    - filter_by_missing(): Filter samples and SNPs based on missing rate thresholds.
    - impute_missing(): Unified imputation entrypoint.
    - grouped_imputation(): Group-based imputation by external labels.
    - _fill_mode(), _fill_mean(), _fill_knn(), etc.: Implement various imputation strategies.

说明：
    本模块是基因型数据管线的核心部分（Preprocessing Layer），
    负责缺失值检测、数据过滤及多算法填补。
    其输出结果直接供降维（embedding）、聚类（clustering）等模块使用。
Description:
    This module forms the preprocessing layer of the genotype data pipeline.
    It handles missing-value detection, filtering, and imputation with multiple algorithms.
    Its outputs are directly consumed by embedding and clustering components.
"""

import json
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA, IncrementalPCA
from joblib import Parallel, delayed
import multiprocessing
import warnings
from tqdm import tqdm
import psutil, time
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def align_by_id(ids: pd.Series, X: pd.DataFrame, meta: pd.DataFrame, id_col: str = "Genetic ID") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按样本 ID 对齐基因型矩阵与注释表，仅保留两表共有的样本，并保持行顺序一致。
    Align genotype matrix and metadata table by sample ID,
    keeping only the intersecting samples and preserving row order.

    :param ids: pd.Series
        样本 ID 序列（与 X 的行一一对应）。
        Series of sample IDs (aligned with rows of X).

    :param X: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :param meta: pd.DataFrame
        样本注释表，包含样本 ID 以及标签信息（如 Y haplogroup）。
        Metadata table containing sample IDs and associated attributes (e.g., Y haplogroup).

    :param id_col: str, default="Genetic ID"
        注释表中的样本 ID 列名（默认 "Genetic ID"）。
        Column name of the sample ID in the metadata table (default: "Genetic ID").

    :return: (X_aligned, meta_aligned)
        - X_aligned: 对齐后的基因型矩阵，仅保留共有样本，索引从 0 重新开始。
          Aligned genotype matrix with only intersecting samples, reindex from 0.
        - meta_aligned: 与 X_aligned 行顺序完全一致的注释表。
          Metadata table aligned in the same row order as X_aligned.

    说明 / Notes:
        - 若注释表中缺失某些样本，将被自动丢弃，仅保留交集。
        - If some samples in metadata are missing, they are automatically discarded — only the intersection is kept.
    """
    print("[INFO] Aligning genotype matrix and metadata by sample ID...")

    # === Step 1: 统一类型，防止类型不匹配 ===
    ids = ids.astype(str)
    meta_ids = meta[id_col].astype(str)

    # === Step 2: 计算交集 ===
    common_ids = pd.Index(ids).intersection(meta_ids)
    if len(common_ids) == 0:
        raise ValueError("[ERROR] No overlapping sample IDs between X and metadata.")

    # === Step 3: 按交集保留样本并保持顺序 ===
    X_aligned = X.loc[ids.isin(common_ids)].reset_index(drop=True)
    meta_aligned = meta.set_index(id_col).loc[common_ids].reset_index()

    print(f"[OK] Matched {len(common_ids)} samples (from {len(ids)} → {len(common_ids)}).")

    # === Step 4: 一致性检查 ===
    if len(X_aligned) != len(meta_aligned):
        print(f"[WARN] Row count mismatch after alignment: X={len(X_aligned)} vs meta={len(meta_aligned)}")
    else:
        print("[CHECK] Alignment successful. Row counts match exactly.")

    return X_aligned, meta_aligned


def compute_missing_rates(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    计算缺失率（样本维度 & SNP 维度）
    Compute missing rates across both sample (row) and SNP (column) dimensions.

    编码说明 / Encoding legend:
        - 0 = 参考等位基因 (reference allele)
        - 1 = 变异等位基因 (alternate allele)
        - 3 = 缺失值 (missing value)

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :return: (sample_missing, snp_missing)
        - sample_missing: 每个样本（行）的缺失率 (0~1)。
          Missing rate of each sample (row-level).
        - snp_missing: 每个 SNP（列）的缺失率 (0~1)。
          Missing rate of each SNP (column-level).

    说明 / Notes:
        - 本函数将编码值 3 替换为 NaN，用于统计缺失率；
          The function replaces code value 3 with NaN for missing-value detection.
        - 使用分块处理，避免一次性生成巨大布尔矩阵导致内存占用过高；
          It processes data in column batches to prevent excessive memory usage.
        - 输出包含两个 Series，分别表示样本维度与 SNP 维度的缺失比例。
          Returns two Series objects: one for sample-level and one for SNP-level missing rates.
    """
    print("[INFO] Compute missing rate:")
    # 将编码 3 替换为 NaN 以便统计缺失率
    Z = X.astype("float32", copy=True)

    # 分块处理，避免生成巨大布尔矩阵
    block_size = 2000  # 每次处理2000列，可根据内存调整
    n_cols = Z.shape[1]

    print(f"[INFO] Replacing missing value (3 → NaN) in {n_cols} columns (batch size = {block_size})")

    for start in tqdm(range(0, n_cols, block_size), desc="Replacing values", ncols=100):
        end = min(start + block_size, n_cols)
        block = Z.iloc[:, start:end]
        mask = block == 3
        block[mask] = np.nan
        Z.iloc[:, start:end] = block

    # 计算每行（样本）与每列（SNP）的缺失率
    sample_missing = Z.isna().mean(axis=1)
    snp_missing = Z.isna().mean(axis=0)
    sample_missing.name = "sample_missing_rate"
    snp_missing.name = "snp_missing_rate"
    print("[OK] Computed the missing rate of each sample and SNP.")
    return sample_missing, snp_missing


def filter_by_missing(X: pd.DataFrame, sample_missing: pd.Series, snp_missing: pd.Series, max_sample_missing: float = 0.8, max_snp_missing: float = 0.8) -> Tuple[pd.DataFrame, pd.Series]:
    """
    按缺失率阈值过滤样本与 SNP（默认阈值较宽松，可在后续调整）。
    Filter samples and SNPs based on missing rate thresholds (default thresholds are lenient and can be adjusted later).

    :param X: pd.DataFrame
        基因型矩阵。
        Genotype matrix.

    :param sample_missing: pd.Series
        每个样本的缺失率。
        Missing rate of each sample.

    :param snp_missing: pd.Series
        每个 SNP 的缺失率。
        Missing rate of each SNP.

    :param max_sample_missing: float, default=0.8
        样本级最大缺失率阈值（默认 0.8）。
        Maximum allowed missing rate per sample (default: 0.8).

    :param max_snp_missing: float, default=0.8
        SNP 级最大缺失率阈值（默认 0.8）。
        Maximum allowed missing rate per SNP (default: 0.8).

    :return: pd.DataFrame
        过滤后的矩阵，索引从 0 重新开始。
        Filtered matrix with reindexed rows starting from 0.

    说明 / Notes:
        - 超过指定缺失率阈值的样本和 SNP 将被剔除。
          Samples or SNPs exceeding the defined thresholds will be removed.
        - 若过滤结果为空矩阵，将抛出异常提醒用户调整阈值。
          If the filtered matrix is empty, an exception will be raised to prompt threshold adjustment.
    """
    print("[INFO] Filter by missing rate:")
    keep_rows = sample_missing <= max_sample_missing
    keep_cols = snp_missing[snp_missing <= max_snp_missing].index
    X_filtered = X.loc[keep_rows, keep_cols].reset_index(drop=True)
    print(f"[INFO] 样本: {keep_rows.sum()}/{len(X)} 保留, SNP: {len(keep_cols)}/{X.shape[1]} 保留")

    if X_filtered.empty:
        raise ValueError("[ERROR] The filtered matrix is empty. Please adjust the threshold.")

    return X_filtered, keep_rows


def filter_meta_by_rows(meta, keep_rows, label="meta"):
    """
    根据 keep_rows 掩码过滤元数据，并打印详细日志。
    Filter metadata by keep_rows mask with logging.

    :param meta: pd.DataFrame
        元数据表，行数与基因矩阵一致。
    :param keep_rows: pd.Series[bool]
        来自 filter_by_missing() 的布尔掩码。
    :param label: str
        日志名称标识（默认 "meta"）。
    :return: pd.DataFrame
        过滤后的元数据。
    """
    total = len(meta)
    kept = keep_rows.sum()
    removed = total - kept

    print(f"[INFO] Filtering {label}:")
    print(f"       - Total rows    : {total}")
    print(f"       - Kept rows     : {kept} ({kept/total:.2%})")
    print(f"       - Removed rows  : {removed} ({removed/total:.2%})")

    # 执行过滤
    meta_filtered = meta.loc[keep_rows].reset_index(drop=True)

    # 一致性检查
    if len(meta_filtered) != kept:
        raise ValueError(
            f"[ERROR] {label} filtering mismatch: "
            f"expected {kept}, got {len(meta_filtered)}"
        )

    print(f"[OK] {label} filtering completed.\n")
    return meta_filtered


def _fill_mode(Z: pd.DataFrame, fallback: float = 1, save_disk: bool = True) -> pd.DataFrame:
    """
    列众数填补（自动内存检测与分片写入优化版）
    Column-wise mode imputation with adaptive memory check and sharded Parquet output.

    本函数根据数据规模与可用内存自动选择填补模式：
    - 当数据规模较小或内存充足时，在内存中直接执行列众数填补；
    - 当数据过大或内存不足时，采用分片（sharded）写入模式，将每批列结果写入 Parquet 文件；
    - 所有中间与最终文件路径均自动管理：临时结果保存在 `data/processed`，最终输出保存在 `data/results`。
    This function automatically chooses the optimal imputation mode:
    - For small datasets or sufficient memory, imputation is performed entirely in-memory;
    - For large datasets or limited memory, a sharded Parquet writing strategy is used to process column batches;
    - Intermediate and final output directories are automatically handled (`data/processed` and `data/results`).

    :param Z: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :param fallback: float
        当整列均为空时使用的回退值。
        Fallback value used when an entire column is empty.

    :param save_disk: bool, default=True
        是否在超大规模数据时启用磁盘写入模式。
        Whether to enable on-disk mode for extremely large datasets.

    :return: pd.DataFrame
        填补后的矩阵。若启用落盘模式，则返回空 DataFrame（结果已保存为 Parquet 文件）。
        The imputed DataFrame; returns an empty DataFrame if on-disk mode is enabled (results saved as Parquet files).

    说明 / Notes:
        - 函数首先估算数据大小（行 × 列 × 4 字节）与系统可用内存，用于判断是否需落盘处理；
          The function estimates dataset size (rows × columns × 4 bytes) and compares it to available system memory to decide whether disk mode is needed.
        - 小规模数据使用字典暂存每列填补结果，并通过 `pd.concat()` 一次性拼接，避免碎片化；
          For small datasets, each column is filled in memory using a dictionary and concatenated with `pd.concat()` to avoid fragmentation.
        - 大规模数据则按列分片（每批约 2000 列）循环处理，并将结果以压缩格式写入 Parquet 文件；
          For large datasets, columns are processed in shards (≈2000 columns per batch) and written as compressed Parquet files.
        - 同时保存列索引元数据（JSON 文件）以便后续加载；
          Column index metadata is also saved as a JSON file for later reconstruction.
        - 若启用了磁盘模式，函数结束后会输出提示路径与加载方式。
          When disk mode is active, the function logs the output directory and instructions for reloading results.
    """
    n_rows, n_cols = Z.shape
    est_size_gb = n_rows * n_cols * 4 / 1024**3
    free_mem_gb = psutil.virtual_memory().available / 1024**3

    print(f"[INFO] Mode filling started: {n_rows}×{n_cols} | est={est_size_gb:.1f} GB | free_mem={free_mem_gb:.1f} GB")

    # === 路径准备 ===
    root_dir = Path(__file__).resolve().parents[2]  # 项目根目录
    processed_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "data" / "results"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    shard_dir = processed_dir / f"mode_filled_{timestamp}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # === 判定是否使用磁盘模式 ===
    use_disk = save_disk and (est_size_gb > free_mem_gb * 0.6 or est_size_gb > 8)
    # === 小规模数据：在内存中直接填补 ===
    if not use_disk:
        print(f"[INFO] Using in-memory mode filling (dataset size={est_size_gb:.1f} GB).")

        filled_cols = {}  # 用字典暂存所有填补结果
        for col in tqdm(Z.columns, desc="Mode filling", ncols=100):
            col_data = Z[col]
            if col_data.isna().all():
                mode_val = fallback
            else:
                mode_val = col_data.mode(dropna=True).iloc[0]
            filled_cols[col] = col_data.fillna(mode_val)

        # 一次性拼接，避免碎片化
        result = pd.concat(filled_cols, axis=1)
        result.columns = Z.columns
        result.index = Z.index
        print(f"[OK] In-memory mode filling complete.")
        return result

    # === 大规模数据，启用磁盘写入（列分片方案） ===
    print(f"[AUTO] Large dataset detected — using sharded Parquet write mode → {shard_dir}")
    batch_cols = 2000
    num_parts = (n_cols + batch_cols - 1) // batch_cols
    # 创建索引列表
    col_index_meta = []

    for part_id, start in enumerate(tqdm(range(0, n_cols, batch_cols), desc="Mode filling (sharded)", ncols=100)):
        end = min(start + batch_cols, n_cols)
        block = Z.iloc[:, start:end]

        # 计算分片众数
        mode_vals = []
        for c in block.columns:
            s = block[c]
            mv = s.mode(dropna=True)
            if len(mv) == 0:
                mode_vals.append(float(fallback))
            else:
                mode_vals.append(float(mv.iloc[0]))
        mode_map = dict(zip(block.columns, mode_vals))

        # 填补并写入 Parquet 文件
        block_filled = block.fillna(value=mode_map)
        block_filled.index = block_filled.index.astype(str)
        part_path = shard_dir / f"part_{part_id:03d}.parquet"
        # noinspection PyArgumentList
        table = pa.Table.from_pandas(df=block_filled.astype("float32"), preserve_index=False)
        pq.write_table(table, part_path, compression="zstd")

    #     # === 记录列信息到索引 ===
        col_index_meta.append({
            "part": f"part_{part_id:03d}.parquet",
            "columns": block.columns.tolist()
        })

    # # === 保存列索引元数据 ===
    meta_path = shard_dir / "columns_index.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(col_index_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Sharded parquet written ({num_parts} files).")
    print(f"[OK] Saved in processed directory → {shard_dir}")
    print(f"[HINT] To load: ds.dataset(r'{shard_dir}', format='parquet')")

    return pd.DataFrame()


def _fill_mean(X: pd.DataFrame) -> pd.DataFrame:
    """
    列均值填补（矢量化实现）
    Column-wise mean imputation (vectorized implementation).

    使用每列的均值替代缺失值，适合连续型数值数据。
    Replaces missing values (NaN) in each column with the column mean, suitable for continuous data.

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :return: pd.DataFrame
        填补后的矩阵，结构与原矩阵完全一致。
        The imputed DataFrame with the same structure as the original.

    说明 / Notes:
        - 对每列单独计算均值，并用 `fillna()` 替换缺失值。
          Computes the mean of each column and replaces NaN values using `fillna()`.
        - 采用 Pandas 内部矢量化操作，无需循环，执行效率高且不会触发碎片警告。
          Utilizes Pandas' internal vectorized operations — faster and avoids fragmentation warnings.
        - 适用于连续数值型特征，不推荐用于类别型或离散型 SNP 编码数据。
          Recommended for continuous numeric features; not ideal for categorical or discrete SNP-encoded data.
    """
    # pandas fillna 本身矢量化，不会触发碎片警告
    result = X.fillna(X.mean())
    result.columns = X.columns
    result.index = X.index
    return result


def _fill_knn(X: pd.DataFrame, n_neighbors: int = 5, metric: str = "nan_euclidean") -> pd.DataFrame:
    """
    基于 sklearn 的 KNN 填补（欧氏距离）
    KNN-based imputation using sklearn with Euclidean distance.

    使用 `sklearn.impute.KNNImputer` 按样本间欧氏距离（忽略 NaN）进行缺失值填补。
    Missing values are imputed using `sklearn.impute.KNNImputer`, which computes pairwise Euclidean distances while ignoring NaN values.

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :param n_neighbors: int, default=5
        近邻数量，用于计算 KNN 填补。
        Number of nearest neighbors used for imputation.

    :param metric: str, default="nan_euclidean"
        距离度量方式（默认忽略 NaN 的欧氏距离）。
        Distance metric to use (default: Euclidean distance ignoring NaNs).

    :return: pd.DataFrame
        填补后的矩阵，结构与原矩阵一致。
        The imputed DataFrame with the same structure as the input.

    说明 / Notes:
        - 使用 sklearn 内置的 `KNNImputer` 实现 KNN 填补。
          Uses sklearn's built-in `KNNImputer` for KNN-based imputation.
        - 在计算样本间距离时会自动忽略缺失值（NaN）。
          Automatically ignores NaN values during distance computation.
        - 当样本量超过 5000 时，会发出性能警告以提示潜在计算开销。
          Prints a performance warning when dataset size exceeds 5000 samples due to computational cost.
    """
    if X.shape[0] > 5000:
        print(f"[WARN] Large dataset detected ({X.shape[0]} samples) — KNN imputation may be slow.")

    imputer = KNNImputer(n_neighbors=n_neighbors, metric=metric)
    M = imputer.fit_transform(X)
    print(f"[OK] sklearn.KNNImputer complete — metric={metric}, k={n_neighbors}")
    return pd.DataFrame(M, columns=X.columns, index=X.index)


def _fill_knn_hamming_abs(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, strategy: str = "mode", random_state: int | None = None) -> pd.DataFrame:
    """
    Hamming(abs) 等权 KNN 填补
    Equal-weight KNN imputation based on unnormalized Hamming distance.

    使用未归一化的汉明距离计算样本间相似度，对缺失值进行 KNN 填补。
    性能相对较慢，仅推荐在小样本数据上使用。
    Performs KNN-based imputation using absolute (unnormalized) Hamming distance to measure sample similarity.
    This method is slower and mainly recommended for small datasets.

    :param X: pd.DataFrame
        基因型矩阵（取值 ∈ {0,1,NaN}）。
        Genotype matrix with values in {0, 1, NaN}.

    :param n_neighbors: int, default=5
        近邻数量。
        Number of nearest neighbors used for imputation.

    :param fallback: float, default=1.0
        当所有邻居均缺失时使用的回退值。
        Fallback value when all neighbors have missing values.

    :param strategy: str, default="mode"
        填补策略，可选："mode"、"round"、"mean"、"prob"。
        Imputation strategy: "mode", "round", "mean", or "prob".
        - mode：众数（最常见值）；
          Uses the most frequent value (mode).
        - round：均值取整；
          Rounds the mean of neighbors.
        - mean：直接取均值；
          Takes the arithmetic mean of neighbors.
        - prob：基于均值概率的二项采样。
          Uses Bernoulli sampling with mean value as probability.

    :param random_state: int | None
        随机种子，仅在 "prob" 策略下生效。
        Random seed, only effective when using "prob" strategy.

    :return: pd.DataFrame
        填补后的矩阵，结构与输入一致。
        The imputed DataFrame with the same structure as the input.

    说明 / Notes:
        - 先将 NaN 替换为占位符 (-1)，以便计算汉明距离。
          NaN values are first replaced by a dummy value (-1) for Hamming distance computation.
        - 使用 `NearestNeighbors(metric="hamming")` 获取最近邻索引与距离。
          Nearest neighbors are computed using sklearn's `NearestNeighbors` with Hamming distance.
        - 距离值乘以特征数以恢复“未归一化”距离度量。
          Distances are multiplied by the number of features to obtain unnormalized values.
        - 对每个缺失值，根据策略从邻居中聚合或采样得到填补值。
          For each missing entry, the imputed value is derived from neighbors according to the chosen strategy.
        - 当无可用邻居时，使用 fallback 值进行替代。
          When no valid neighbors are available, the fallback value is used instead.
    """
    X = X.copy()
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    # sklearn 的 Hamming 不能处理 NaN，先替换为哑值
    X_tmp = X.fillna(-1)

    # 构建最近邻模型（Hamming 距离）
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="hamming")
    nbrs.fit(X_tmp)

    # 获取最近邻（包括自身）
    distances, indices = nbrs.kneighbors(X_tmp)
    distances *= n_features # 转换为“未归一化汉明距离”
    indices = indices[:, 1:] # 去掉自身

    # 执行填补
    for i in tqdm(range(n_samples), desc="Filling samples", ncols=100):
        for j in range(n_features):
            if pd.isna(X.iat[i, j]):
                vals = X.iloc[indices[i], j]
                mask = ~vals.isna()
                if mask.any():
                    if strategy == "mode":
                        X.iat[i, j] = vals[mask].mode().iat[0]
                    elif strategy == "round":
                        X.iat[i, j] = round(vals[mask].mean())
                    elif strategy == "prob":
                        p = vals[mask].mean()
                        X.iat[i, j] = rng.binomial(1, p)
                    elif strategy == "mean":
                        X.iat[i, j] = vals[mask].mean()
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                else:
                    X.iat[i, j] = fallback
    print(f"[OK] Hamming(abs) KNN complete — k={n_neighbors}, strategy={strategy}, fallback={fallback}, seed={random_state}")
    return X


def _fill_knn_hamming_adaptive(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, target_decay: float = 0.5, strategy: str = "mode", log_samples: int = 3, random_state: int | None = None, parallel: bool = True) -> pd.DataFrame:
    """
    自适应加权汉明 KNN 填补（Adaptive Weighted Hamming KNN）
    Adaptive weighted KNN imputation using exponential decay on Hamming distance.

    本函数使用自适应衰减因子 α 对邻居距离进行加权，
    在保持高精度的同时显著提升填补效率。
    支持向量化与多核并行，适用于高维稀疏的基因型矩阵。
    This function applies an adaptive decay factor α to neighbor distances for weighted imputation,
    improving both accuracy and speed. It supports vectorized operations and multi-core parallelism,
    making it suitable for high-dimensional sparse genotype matrices.

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :param n_neighbors: int, default=5
        近邻数量。
        Number of nearest neighbors.

    :param fallback: float, default=1.0
        若所有邻居均缺失时的回退值。
        Fallback value used when all neighbors are missing.

    :param target_decay: float, default=0.5
        控制权重中位邻居的衰减比例。
        Controls the weight decay for the median-distance neighbor.

    :param strategy: str, default="mode"
        填补策略，可选："mode"、"round"、"mean"、"prob"。
        Imputation strategy: "mode", "round", "mean", or "prob".
        - mode：基于加权投票的众数填补（推荐）。
          Weighted majority vote (recommended).
        - round：对加权均值四舍五入取整。
          Rounds the weighted mean.
        - mean：直接使用加权均值（连续输出）。
          Uses weighted mean directly (continuous output).
        - prob：按加权均值作为概率进行二项采样。
          Bernoulli sampling based on weighted mean probability.

    :param log_samples: int, default=3
        控制打印日志的样本数量（输出 α 与前几个权重）。
        Number of sample logs to display (prints α and top neighbor weights).

    :param random_state: int | None
        随机种子，用于 "prob" 策略的可重复性。
        Random seed, used for reproducibility when using the "prob" strategy.

    :param parallel: bool, default=True
        是否启用列级并行（使用 Joblib）。
        Whether to enable column-level parallelism via Joblib.

    :return: pd.DataFrame
        填补后的矩阵，与原输入结构相同。
        The imputed DataFrame with the same structure as the input.

    说明 / Notes:
        - 首先将 NaN 替换为 -1，并使用 BallTree 基于汉明距离计算邻居。
          NaN values are first replaced with -1, and BallTree is used to find neighbors via Hamming distance.
        - 计算每个样本的距离中位数并自适应生成 α，随后按指数衰减计算权重矩阵。
          For each sample, α is derived from its median distance, and weights are computed via exponential decay.
        - 并行模式下，每列的填补任务使用 Joblib 分发至多核处理器执行。
          In parallel mode, column-level tasks are distributed across multiple CPU cores via Joblib.
        - 每列填补函数根据所选策略（mode/mean/round/prob）独立执行缺失值估算。
          Each column’s imputation is performed independently based on the selected strategy.
        - 随机选择 log_samples 个样本打印 α、距离与前 5 个邻居权重，以便验证权重分布合理性。
          Randomly selected samples log α, distance, and top-5 neighbor weights for interpretability.
        - 高效、可扩展，适合中大规模 SNP 矩阵的快速加权填补。
          Efficient and scalable; suitable for medium-to-large SNP matrices with adaptive weighting.
    """
    X = X.copy()
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    # === 替换 NaN, 构建 BallTree, 并计算未归一化汉明距离===
    X_tmp = X.fillna(-1).to_numpy()
    tree = BallTree(X_tmp, metric="hamming")  # type: ignore[arg-type]
    dist, ind = tree.query(X_tmp, k=n_neighbors + 1)
    dist, ind = dist[:, 1:], ind[:, 1:]  # 去掉自身
    dist *= n_features  # 未归一化汉明距离

    # === 向量化计算 α & 权重矩阵 ===
    median_d = np.median(dist, axis=1, keepdims=True)
    alpha = np.log(1 / target_decay) / np.maximum(median_d, 1e-9)
    weights = np.exp(-alpha * dist)

    # 随机打印部分样本日志
    log_idx = rng.choice(n_samples, size=min(log_samples, n_samples), replace=False)

    # for i in log_idx:
    #     print(f"[INFO] Sample #{i:<4d} | α={alpha[i,0]:.4f} | median_d={median_d[i,0]:.2f} | mean_d={dist[i].mean():.2f}")

    for i in log_idx:
        # 打印前 5 个邻居权重（保留两位小数）
        w_preview = np.round(weights[i, :5], 3)
        print(f"[INFO] Sample #{i:<4d} | α={alpha[i, 0]:.4f} | median_d={median_d[i, 0]:.2f} | "
              f"mean_d={dist[i].mean():.2f} | w[:5]={w_preview}")

    # === 定义单列填补函数 ===
    def _process_column(j: int) -> np.ndarray:
        """
        对单列执行缺失值填补逻辑（可并行）。
        ---------------------------------------------------
        :param j: int 列索引。

        :return: col: np.ndarray 该列填补后的完整列向量。

        逻辑:
            1. 找出该列中所有缺失样本；
            2. 对每个缺失样本，根据其 k 个邻居进行加权计算；
            3. 根据 strategy 选择不同的填补方式；
            4. 若邻居均缺失，则使用 fallback。
        """
        # 拷贝该列的原始数值（NaN 已替换为 -1）
        col = X_tmp[:, j].copy()

        # 找出该列缺失的行索引（原始 DataFrame 判断 NaN）
        missing_rows = np.where(np.isnan(X.iloc[:, j].to_numpy()))[0]
        if len(missing_rows) == 0:
            return col # 若该列无缺失值则直接返回

        # 遍历所有缺失行，逐样本填补
        for row_idx in missing_rows:
            vals = X_tmp[ind[row_idx], j]     # 当前样本 i 的近邻在第 j 列的取值
            mask = vals != -1           # 过滤掉缺失邻居
            if not np.any(mask):        # 若所有邻居都缺失
                col[row_idx] = fallback
                continue
            v, w = vals[mask], weights[row_idx, mask]  # 有效邻居值及对应权重
            weighted_mean = np.average(v, weights=w)

            # === 根据不同策略执行填补 ===
            if strategy == "mode":
                # 权重众数（通过加权投票）
                val0 = np.sum(w[v == 0])
                val1 = np.sum(w[v == 1])
                col[row_idx] = 1.0 if val1 >= val0 else 0.0
            elif strategy == "round":
                # 四舍五入加权均值
                col[row_idx] = np.rint(weighted_mean)
            elif strategy == "prob":
                # 概率采样（按加权均值作为概率）
                col[row_idx] = rng.binomial(1, weighted_mean)
            elif strategy == "mean":
                # 连续加权均值
                col[row_idx] = weighted_mean
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        return col

    # === 并行或串行执行 ===
    if parallel and n_features > 500:
        # 获取当前 CPU 核心数，动态设定并行任务数量
        n_jobs = multiprocessing.cpu_count()
        print(f"[INFO] Adaptive KNN parallel mode — {n_features} columns, jobs={n_jobs}")

        # 使用 Joblib 并行执行每列填补
        # delayed(_process_column)(j) 表示并行调用列级函数
        new_cols = Parallel(n_jobs=n_jobs)(
            delayed(_process_column)(j) for j in range(n_features)
        )

        # 将结果按列堆叠（列顺序与原始矩阵保持一致）
        X_filled = pd.DataFrame(np.column_stack(new_cols), columns=X.columns)
    else:
        # 串行模式：逐列依次执行填补
        # 仅在该列存在 NaN 时才调用填补函数，避免不必要的计算
        for j in tqdm(range(n_features), desc="Filling columns", ncols=100):
            if X.iloc[:, j].isna().any():
                X_tmp[:, j] = _process_column(j)

        # 将填补后的 numpy 数组重新封装为 DataFrame
        X_filled = pd.DataFrame(X_tmp, columns=X.columns)

    print(f"[OK] Adaptive Weighted KNN (fast) complete — k={n_neighbors}, strategy={strategy}, parallel={parallel}")
    return X_filled


def _fill_knn_hamming_balltree(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, strategy: str = "mode", parallel: bool = False, n_jobs: int = -1) -> pd.DataFrame:
    """
    基于 BallTree 的加速 Hamming KNN 填补
    Accelerated Hamming KNN imputation using BallTree index.

    使用 BallTree 空间索引在汉明距离下进行高效近邻搜索，
    支持列级并行和向量化操作。相比传统 KNN 填补，
    该方法速度更快、内存占用更低，适合中等规模数据集。
    Uses BallTree spatial indexing to accelerate Hamming distance neighbor search.
    Supports column-level parallelism and vectorized computation.
    Faster and more memory-efficient than traditional KNN imputation,
    suitable for medium-scale datasets.

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :param n_neighbors: int, default=5
        近邻数量。
        Number of nearest neighbors.

    :param fallback: float, default=1.0
        若所有邻居均缺失时的回退值。
        Fallback value when all neighbors have missing values.

    :param strategy: str, default="mode"
        填补策略，可选："mode"、"mean"、"round"。
        Imputation strategy: "mode", "mean", or "round".
        - mode：统计邻居中出现频率最高的值（众数）。
          Uses the most frequent value (mode) among neighbors.
        - mean：计算邻居的均值作为填补值。
          Takes the arithmetic mean of neighbors.
        - round：对均值进行四舍五入。
          Rounds the mean to the nearest integer.

    :param parallel: bool, default=False
        是否启用列级并行（Joblib）。
        Whether to enable column-level parallelism via Joblib.

    :param n_jobs: int, default=-1
        并行任务数（默认使用所有 CPU 核心）。
        Number of parallel jobs (default: use all CPU cores).

    :return: pd.DataFrame
        填补后的矩阵，结构与输入一致。
        The imputed DataFrame with the same structure as the input.

    说明 / Notes:
        - 使用 BallTree 在 Hamming 距离空间中高效搜索最近邻，性能优于暴力搜索。
          BallTree efficiently searches nearest neighbors under Hamming distance, outperforming brute-force search.
        - 小数据集时会自动禁用并行，以避免进程创建开销。
          Automatically disables parallel mode for small datasets to reduce multiprocessing overhead.
        - 每列独立执行填补操作，可串行或并行运行。
          Each column is processed independently and can run in serial or parallel mode.
        - 对缺失样本，提取邻居值并根据策略执行统计（众数或均值）。
          For missing samples, neighbor values are aggregated according to the chosen strategy (mode or mean).
        - 若所有邻居均为缺失值，则使用 fallback 值进行填充。
          When all neighbor values are missing, the fallback value is used.
        - 推荐用于中等规模数据集（约 10⁴~10⁵ 个样本 × SNP 特征）。
          Recommended for medium-scale datasets (~10⁴–10⁵ samples × SNP features).
    """
    X = X.copy()
    n_samples, n_features = X.shape
    X_tmp = X.fillna(-1).to_numpy()

    # 小数据集时禁用并行（避免进程创建开销）
    threshold = 2e6  # 元素数量阈值
    if parallel and (n_samples * n_features < threshold):
        print(f"[AUTO] Data too small ({n_samples}×{n_features}), disabling parallel for efficiency.")
        parallel = False

    # 构建 BallTree 并查询最近邻索引
    tree = BallTree(X_tmp, metric="hamming")  # type: ignore[arg-type]
    dist, ind = tree.query(X_tmp, k=n_neighbors + 1)
    ind = ind[:, 1:]  # 去掉自身

    def _process_column(j: int) -> np.ndarray:
        """
        对单列执行缺失值填补逻辑（可并行）。
        ---------------------------------------------------
        :param j: int 列索引。

        :return: col: np.ndarray 该列填补后的完整列向量。

        逻辑:
            1. 找出该列中所有缺失样本；
            2. 对每个缺失样本提取其邻居；
            3. 根据 strategy 选择不同的填补方式；
            4. 若邻居均缺失，则使用 fallback。
        """
        # 对每个缺失样本提取其邻居
        col_values = X_tmp[:, j].copy()
        new_col = col_values.copy()

        # 找出该列中缺失的样本索引
        missing_rows = np.where(np.isnan(X.iloc[:, j].to_numpy()))[0]
        if len(missing_rows) == 0:
            return new_col

        # 获取这些缺失行的邻居矩阵（形状 = [缺失数, k]）
        neighbor_matrix = X_tmp[ind[missing_rows], j]  # shape = (num_missing, k)

        # === “众数”策略 ===
        if strategy == "mode":
            for idx, vals in enumerate(neighbor_matrix):
                valid = vals[vals != -1] # 有效邻居
                if valid.size == 0:
                    new_col[missing_rows[idx]] = fallback
                else:
                    # 统计邻居值出现次数，取最多的作为填充值
                    uniq, counts = np.unique(valid, return_counts=True)
                    new_col[missing_rows[idx]] = uniq[np.argmax(counts)]

        # === “均值/四舍五入”策略 ===
        elif strategy in ("mean", "round"):
            # 将缺失邻居标记为 NaN，计算行均值
            means = np.where(neighbor_matrix != -1, neighbor_matrix, np.nan)
            means = np.nanmean(means, axis=1)
            means[np.isnan(means)] = fallback  # 若全缺失则填 fallback
            new_col[missing_rows] = np.rint(means) if strategy == "round" else means
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return new_col

    # === 并行或串行执行列级填补 ===
    if parallel:
        print(f"[INFO] Parallel BallTree filling — {n_features} columns, jobs={n_jobs}")

        # 使用 Joblib 并行地处理所有列
        # 每列调用一次 _process_column()，各进程独立运行
        new_cols = Parallel(n_jobs=n_jobs)(
            delayed(_process_column)(j) for j in range(n_features)
        )

        # 将所有处理好的列组合成新的 DataFrame
        X_filled = pd.DataFrame(np.column_stack(new_cols), columns=X.columns)
    else:
        # 串行执行模式（单线程）
        # 遍历每一列，仅在存在 NaN 时才调用填补函数
        for j in tqdm(range(n_features), desc="Filling columns", ncols=100):
            if X.iloc[:, j].isna().any():
                X_tmp[:, j] = _process_column(j)

        # 将填补完成的矩阵重新封装为 DataFrame
        X_filled = pd.DataFrame(X_tmp, columns=X.columns)

    print(f"[OK] Hamming(BallTree-fast) KNN complete — k={n_neighbors}, strategy={strategy}, parallel={parallel}")
    return X_filled


def _fill_knn_faiss(X: pd.DataFrame, n_neighbors: int = 5, n_components: int = 50, fallback: float = 1.0, strategy: str = "mode", random_state: int | None = 42, batch_size: int = 2000) -> pd.DataFrame:
    """
    PCA + Faiss 近似 KNN 填补（增强版，支持 IncrementalPCA）  (未验证)
    PCA + Faiss approximate KNN imputation (enhanced version with IncrementalPCA support).  (invalid)

    该函数专为大规模基因型矩阵（>10k 样本）设计：
    先通过 PCA/IncrementalPCA 降维压缩特征空间，再使用 Faiss 实现高速 KNN 搜索。
    自动在 CPU 与 GPU 间切换，可在有限内存环境下高效完成缺失值填补。
    This function is designed for large-scale genotype matrices (>10k samples):
    It first performs dimensionality reduction via PCA or IncrementalPCA,
    then uses Faiss for fast approximate nearest-neighbor search (CPU/GPU supported).
    The method adapts automatically to available memory, ensuring robust imputation on big data.

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP；值 ∈ {0, 1, NaN}）。
        Genotype matrix (rows = samples, columns = SNPs; values ∈ {0, 1, NaN}).

    :param n_neighbors: int, default=5
        KNN 近邻数量。
        Number of nearest neighbors used for imputation.

    :param n_components: int, default=50
        PCA 降维后的特征维度数。
        Number of components retained after PCA reduction.

    :param fallback: float, default=1.0
        当所有邻居均缺失时使用的回退值。
        Fallback value used when all neighbors are missing.

    :param strategy: str, default="mode"
        填补策略，可选："mode"、"mean"、"round"。
        Imputation strategy: "mode", "mean", or "round".
        - mode：使用邻居的众数。
          Uses the most frequent neighbor value (mode).
        - mean：取邻居均值。
          Takes the arithmetic mean of neighbors.
        - round：对均值进行四舍五入。
          Rounds the mean to the nearest integer.

    :param random_state: int | None, default=42
        随机种子，用于保证可重复性。
        Random seed for reproducibility.

    :param batch_size: int, default=2000
        IncrementalPCA 的批处理大小。
        Batch size for IncrementalPCA.

    :return: pd.DataFrame
        填补后的矩阵（行列与原输入一致）。
        The imputed DataFrame with the same shape as the input.

    说明 / Notes:
        - 当样本数较大时自动启用 IncrementalPCA，以降低内存占用。
          Automatically switches to IncrementalPCA for large datasets to reduce memory usage.
        - PCA 阶段先使用列均值暂时填补 NaN，以确保矩阵可分解。
          NaN values are temporarily filled with column means before PCA decomposition.
        - Faiss 用于在 PCA 降维空间中加速近邻搜索，可显著提升性能。
          Faiss accelerates KNN search in the reduced PCA space for high efficiency.
        - 支持 CPU 版本 (faiss-cpu) 与 GPU 版本 (faiss-gpu)，自动检测环境。
          Supports both CPU (`faiss-cpu`) and GPU (`faiss-gpu`) backends depending on environment.
        - 若 PCA 失败，则自动回退到列众数填补以确保稳定性。
          If PCA fails, the function automatically falls back to column-mode imputation.
        - 推荐用于样本量超过 10k 的大规模基因型数据集。
          Recommended for large-scale genotype datasets with more than 10k samples.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Faiss 未安装，请运行: pip install faiss-cpu 或 faiss-gpu")

    n_samples, n_features = X.shape
    est_gb = n_samples * n_features * 4 / (1024 ** 3)
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"[INFO] PCA+Faiss (safe) | shape={n_samples}×{n_features}, est_mem≈{est_gb:.1f} GB, avail={avail_gb:.1f} GB")

    # === Step 1. 填补临时缺失值 (先用列均值代替 NaN，用于 PCA) ===
    X_filled = X.fillna(X.mean())

    # 自动选择 PCA 版本
    if n_samples > 5000:
        print(f"[INFO] 使用 IncrementalPCA(batch_size={batch_size}) 防止内存爆炸")
        pca = IncrementalPCA(n_components=min(n_components, n_samples - 1), batch_size=batch_size)
    else:
        pca = PCA(n_components=min(n_components, n_samples - 1), random_state=random_state)

    # 执行 PCA 降维
    try:
        Z = pca.fit_transform(X_filled.values)
    except Exception as e:
        print(f"[WARN] PCA 失败 ({e})，改用众数填补。")
        return _fill_mode(X, fallback=fallback)

    # 构建 CPU 版 Faiss 索引
    print("[INFO] 构建 CPU 版 Faiss 索引 ...")
    index = faiss.IndexFlatL2(Z.shape[1])
    index.add(Z.astype("float32"))
    D, I = index.search(Z.astype("float32"), n_neighbors + 1)
    I = I[:, 1:]  # 去掉自身

    # 逐样本填补（单线程，稳定不崩）
    X_np = X.values.copy()
    nan_mask = np.isnan(X_np)
    print(f"[INFO] 开始填补 (strategy={strategy}, fallback={fallback})")

    for i in tqdm(range(X_np.shape[0]), desc="FAISS filling", ncols=100):
        missing_cols = np.where(nan_mask[i])[0]
        if missing_cols.size == 0:
            continue
        neigh_idx = I[i]
        for j in missing_cols:
            vals = X_np[neigh_idx, j]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                X_np[i, j] = fallback
            elif strategy == "mode":
                v, c = np.unique(vals, return_counts=True)
                X_np[i, j] = v[np.argmax(c)]
            elif strategy == "mean":
                X_np[i, j] = np.nanmean(vals)
            elif strategy == "round":
                X_np[i, j] = np.round(np.nanmean(vals))
        if i % 500 == 0:
            print(f"  [Progress] {i}/{n_samples} samples")

    print(f"[OK] PCA+Faiss(SAFE) 填补完成 — k={n_neighbors}, dim={pca.n_components_}")
    return pd.DataFrame(X_np, index=X.index, columns=X.columns)


def _fill_knn_auto(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, missing_threshold: float = 0.4, random_state: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    自动选择最优 KNN 填补算法（Auto-select KNN Imputation）
    Automatically selects the optimal KNN imputation strategy based on dataset scale and missing rate.

    根据样本规模、缺失率与内存估算值，自动选择最合适的 KNN 填补算法：
    小数据使用精确 Hamming，较大数据使用 BallTree 或自适应加权 Hamming，
    超大数据集自动切换至 PCA + Faiss 近似 KNN 填补。
    Automatically selects the appropriate KNN-based imputation method
    depending on dataset size, missingness ratio, and estimated memory footprint.
    Small datasets use exact Hamming, medium datasets use BallTree or adaptive weighting,
    and very large datasets fall back to PCA + Faiss for approximate KNN.

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :param n_neighbors: int, default=5
        近邻数量。
        Number of nearest neighbors.

    :param fallback: float, default=1.0
        当所有邻居均缺失时使用的回退值。
        Fallback value used when all neighbors are missing.

    :param missing_threshold: float, default=0.4
        缺失率阈值，超过此比例触发高缺失模式。
        Missing rate threshold to trigger high-missingness strategy.

    :param random_state: int | None
        随机种子。
        Random seed for reproducibility.

    :param verbose: bool, default=True
        是否打印算法选择与执行日志。
        Whether to print detailed selection and execution logs.

    :return: pd.DataFrame
        填补后的矩阵，结构与输入一致。
        The imputed DataFrame with the same structure as the input.

    说明 / Notes:
        - 该函数通过自动检测数据规模与缺失率，在多种 KNN 变体间动态选择最优算法。
          Automatically detects dataset scale and missingness to dynamically pick the optimal imputation algorithm.
        - 算法选择逻辑如下：
          The strategy selection logic is as follows:
            1. 估算内存占用 > 20 GB → 列众数填补（`_fill_mode`）
               Estimated memory > 20 GB → use simple column-wise mode imputation.
            2. 缺失率 ≥ 40% → 自适应概率 Hamming（`_fill_knn_hamming_adaptive`, strategy="prob"）
               Missing rate ≥ 40% → adaptive probabilistic Hamming imputation.
            3. n < 500 → 精确 Hamming(abs)（`_fill_knn_hamming_abs`）
               n < 500 → exact Hamming(abs) KNN.
            4. 500 ≤ n ≤ 5000 → BallTree 并行 KNN（`_fill_knn_hamming_balltree`）
               500 ≤ n ≤ 5000 → BallTree-accelerated parallel KNN.
            5. 5000 < n ≤ 10000 → 自适应加权 Hamming（`_fill_knn_hamming_adaptive`）
               5000 < n ≤ 10000 → adaptive weighted Hamming KNN.
            6. n > 10000 → PCA + Faiss（`_fill_knn_faiss`，自动切换 IncrementalPCA）
               n > 10000 → PCA + Faiss approximate KNN with automatic IncrementalPCA fallback.
        - 若 Faiss 或 PCA 失败，函数将自动回退至列众数填补以确保稳定性。
          If Faiss or PCA fails, the function safely falls back to mode-based imputation.
        - 该方法适用于自动化数据管线，可在无需人工干预的情况下选择高效算法。
          Intended for automated preprocessing pipelines, requiring no manual tuning of imputation strategy.
    """
    # === 基础信息 ===
    n_samples, n_features = X.shape
    missing_rate = X.isna().mean().mean()
    est_size_gb = n_samples * n_features * 4 / (1024**3)  # 估算 float32 内存大小

    if verbose:
        print(f"[INFO] Data shape: {n_samples}×{n_features} | Missing rate={missing_rate:.2%} | Est. size={est_size_gb:.1f} GB")

    # ===  极大矩阵 — 仅保留简单众数填补 ===
    if est_size_gb > 20:
        print("[AUTO] Extremely large dataset (>20 GB est.) — using MODE (column-wise majority) imputation.")
        return _fill_mode(X, fallback=fallback)

    # ===  高缺失率（>30%） — 概率自适应 Hamming ===
    if missing_rate >= missing_threshold:
        if verbose:
            print("[AUTO] High missingness detected — using Adaptive Hamming (probabilistic mode).")
        return _fill_knn_hamming_adaptive(
            X,
            n_neighbors=n_neighbors,
            fallback=fallback,
            target_decay=0.4,
            strategy="prob",
            random_state=random_state,
        )

    # ===  小样本 — Hamming(abs) 精确等权 ===
    if n_samples < 500:
        if verbose:
            print("[AUTO] Small dataset detected — using Hamming(abs) (equal-weight mode).")
        return _fill_knn_hamming_abs(
            X,
            n_neighbors=n_neighbors,
            fallback=fallback,
            strategy="mode",
            random_state=random_state,
        )

    # ===  中等规模 — BallTree 并行版本 ===
    elif n_samples <= 5000:
        if verbose:
            print("[AUTO] Medium dataset — using BallTree-fast Hamming KNN.")
        return _fill_knn_hamming_balltree(
            X,
            n_neighbors=n_neighbors,
            fallback=fallback,
            strategy="mode",
            parallel=True,
        )

    # ===  大规模 — 自适应加权 Hamming KNN ===
    elif n_samples <= 10000:
        if verbose:
            print("[AUTO] Large dataset — using Adaptive Weighted Hamming KNN.")
        return _fill_knn_hamming_adaptive(
            X,
            n_neighbors=n_neighbors,
            fallback=fallback,
            target_decay=0.5,
            strategy="mode",
            random_state=random_state,
        )

    # ===  超大规模 — 使用 PCA + Faiss（自动切换 IncrementalPCA） ===
    else:
        if verbose:
            print("[AUTO] Very large dataset — trying PCA + Faiss approximate KNN with IncrementalPCA fallback.")
        try:
            return _fill_knn_faiss(
                X,
                n_neighbors=n_neighbors,
                n_components=50,
                fallback=fallback,
                strategy="mode",
                random_state=random_state,
            )
        except Exception as e:
            print(f"[WARN] Faiss KNN failed ({e}), falling back to MODE imputation.")
            return _fill_mode(X, fallback=fallback)


def impute_missing(X: pd.DataFrame, method: str = "mode", n_neighbors: int = 5) -> pd.DataFrame:
    """
    通用缺失值填补入口函数
    General entry point for missing-value imputation.

    根据输入参数自动选择对应的填补算法，并执行缺失值替换。
    Automatically selects and applies the appropriate imputation method based on input parameters.

    支持方法 / Supported methods:
        - "mode"                 → 列众数填补 / Column-wise mode imputation
        - "mean"                 → 列均值填补 / Column-wise mean imputation
        - "knn"                  → sklearn KNN（欧氏距离） / sklearn KNN (Euclidean distance)
        - "knn_hamming_abs"      → 等权 Hamming KNN / Equal-weight Hamming KNN
        - "knn_hamming_adaptive" → 自适应加权 Hamming KNN / Adaptive weighted Hamming KNN
        - "knn_hamming_balltree" → BallTree 加速 Hamming KNN / BallTree-accelerated Hamming KNN
        - "knn_faiss"            → PCA + Faiss 近似 KNN 填补 / PCA + Faiss approximate KNN
        - "knn_auto"             → 自动策略选择 / Automatic strategy selection

    :param X: pd.DataFrame
        基因型矩阵（0 = 参考等位基因, 1 = 变异等位基因, 3 = 缺失）。
        Genotype matrix (0 = reference allele, 1 = alternate allele, 3 = missing).

    :param method: str, default="mode"
        填补方法（支持上述关键字）。
        Imputation method to apply (see list above).

    :param n_neighbors: int, default=5
        KNN 近邻数量（仅对 KNN 类方法有效）。
        Number of nearest neighbors (applicable to KNN-based methods only).

    :return: pd.DataFrame
        填补后的矩阵，与输入结构一致。
        The imputed DataFrame with the same structure as the input.

    说明 / Notes:
        - 本函数统一管理所有缺失值填补算法的调用逻辑，支持自动日志输出与方法检测。
          This function manages all imputation methods in a unified interface, with automatic logging and method dispatch.
        - 首先将编码值 3 转换为 NaN，以兼容 pandas/numpy 运算。
          Values encoded as `3` are first converted to NaN for compatibility with pandas/numpy operations.
        - 内部通过分块处理避免大矩阵布尔索引导致的内存峰值问题。
          Missing-value replacement is performed in batches to prevent excessive memory usage.
        - 支持从简单统计方法到复杂 KNN/Faiss 算法的多层级策略。
          Supports a hierarchy of methods from simple statistical filling to advanced KNN/Faiss algorithms.
        - 自动打印填补前后缺失数量，并报告残留 NaN 情况。
          Prints the number of missing values before/after imputation and reports remaining NaNs, if any.
        - 若 method 参数无效，将抛出 ValueError。
          Raises ValueError if the given method name is invalid.
    """
    print(f"[INFO] Impute missing values with method: {method}")
    method = method.lower()

    # 将编码3替换为NaN以便填补
    Z = X.astype("float32", copy=True)

    # 分块处理，避免生成巨大布尔矩阵
    block_size = 2000  # 每次处理2000列，可根据内存调整
    n_cols = Z.shape[1]

    print(f"[INFO] Replacing missing value (3 → NaN) in {n_cols} columns (batch size = {block_size})")

    for start in tqdm(range(0, Z.shape[1], block_size), desc="3→NaN replace", ncols=100):
        end = min(start + block_size, n_cols)
        block = Z.iloc[:, start:end].to_numpy()
        block[block == 3] = np.nan
        Z.iloc[:, start:end] = block

    before_nans = Z.isna().sum().sum()

    if method == "mode":
        filled =  _fill_mode(Z)
    elif method == "mean":
        filled =  _fill_mean(Z)
    elif method == "knn":
        filled =  _fill_knn(Z, n_neighbors=n_neighbors)
    elif method == "knn_hamming_abs":
        filled =  _fill_knn_hamming_abs(Z, n_neighbors=n_neighbors, random_state=42)
    elif method == "knn_hamming_adaptive":
        filled =  _fill_knn_hamming_adaptive(Z, n_neighbors=n_neighbors, random_state=42)
    elif method == "knn_hamming_balltree":
        filled = _fill_knn_hamming_balltree(Z, n_neighbors=n_neighbors,parallel=True)
    elif method == "knn_faiss":
        filled = _fill_knn_faiss(Z, n_neighbors=n_neighbors, random_state=42)
    elif method == "knn_auto":
        filled =  _fill_knn_auto(Z, n_neighbors=n_neighbors, random_state=42)
    else:
        raise ValueError(f"Unknown filling method: {method}")

    after_nans = filled.isna().sum().sum()
    replaced = before_nans - after_nans

    if after_nans > 0:
        print(f"[WARN] There are still {after_nans} NaNs after padding (possibly an abnormal column).")
    print(f"[OK] {method.upper():<5} Filling complete | Number of missing values replaced: {replaced} | Residual NaN: {after_nans}")

    return filled


def grouped_imputation(X, labels, method: str = "mode"):
    """
    按外部标签分组执行缺失值填补（封装版）  (未验证)
    Perform missing-value imputation by external group labels (wrapper version).  (invalide)

    根据外部分类标签（如地理区域或单倍群）将样本划分为子集，
    并在每个分组内部独立执行缺失值填补。
    若未提供标签，则执行全局填补。
    Performs missing-value imputation separately for each group defined by an external label
    (e.g., World Zone, Y haplogroup). If no label is provided, applies global imputation.

    :param X: pd.DataFrame
        原始基因型矩阵（行 = 样本，列 = SNP）。
        Original genotype matrix (rows = samples, columns = SNPs).

    :param labels: pd.Series or None
        外部分组标签列，例如世界区域或单倍群分类。
        若为 None，则执行全局填补而不分组。
        External grouping labels (e.g., World Zone or haplogroup).
        If None, global imputation is performed without grouping.

    :param method: str, default="mode"
        缺失值填补方法（如 "mode"、"knn_hamming_adaptive" 等）。
        Imputation method to use (e.g., "mode", "knn_hamming_adaptive", etc.).

    :return: pd.DataFrame
        分组填补后的完整矩阵，与原矩阵索引顺序一致。
        The fully imputed DataFrame, preserving the original index order.

    说明 / Notes:
        - 若未提供标签（labels=None），函数将自动执行全局填补。
          If `labels` is None, a single global imputation will be performed.
        - 每个分组根据标签独立执行填补，分组之间互不影响。
          Each label group is processed independently without affecting others.
        - 样本量过小（≤5）时，自动降级为列众数填补以避免过拟合。
          For small groups (≤5 samples), automatically switches to mode imputation.
        - 若分组过小且方法为 "knn_faiss"，会回退为 "mode" 填补。
          If the group is too small and method is "knn_faiss", falls back to "mode" imputation.
        - 处理完成后将所有分组结果重新合并并按原索引排序。
          After processing, all group results are concatenated and re-indexed in original order.
        - 输出结果适用于保持地理或生物学一致性的局部分布式填补。
          Designed for localized or biologically consistent imputation respecting group structure.
    """
    if labels is None:
        print(f"[INFO] Global imputation (no grouping label provided) ...")
        return impute_missing(X, method=method)

    group_col = labels.name
    print(f"[INFO] Grouped imputation by external label '{group_col}' ...")

    # 保证索引一致
    labels = labels.reindex(X.index)
    groups = list(labels.groupby(labels).groups.items())
    filled_groups = []
    MIN_GROUP_SIZE = 5  # 小于等于 5 个样本时跳过 KNN

    for idx, (group_name, sample_idx) in enumerate(tqdm(groups, desc="Group filling progress", ncols=100, unit="group")):
        sub_X = X.loc[sample_idx]
        n = len(sub_X)
        tqdm.write(f"[INFO] Filling group {idx+1}/{len(groups)}: '{group_name}' (n={n})")

        # 如果样本太少，改用 mode 填补
        if n <= 50 and method.startswith("knn_faiss"):
            print(f"    [WARN] Group '{group_name}' too small (n={n}), switching to mode imputation.")
            sub_X_filled = impute_missing(sub_X, method="mode")
        elif n <= MIN_GROUP_SIZE and method.startswith("knn"):
            print(f"    [WARN] Group '{group_name}' too small (n={n}), using 'mode' instead of {method}")
            sub_X_filled = impute_missing(sub_X, method="mode")
        else:
            sub_X_filled = impute_missing(sub_X, method=method)

        filled_groups.append(sub_X_filled)

    X_filled = pd.concat(filled_groups).sort_index()
    print(f"[OK] Grouped imputation completed for {len(filled_groups)} groups.")
    return X_filled


def clean_labels_and_align(
        emb: pd.DataFrame,
        labels: pd.Series,
        collapse_y: bool = False,
        invalid_values=None,
        whitelist=None,
        
):
    """
    清洗分类标签并与降维后的 embedding 对齐。
    Clean categorical labels and align them with the embedding matrix.

    ==========================
    功能说明 / Functionality
    ==========================
    - 移除无效标签（如 "NA", "n/a", "..", ""）
    - 若设置 whitelist，则仅保留白名单中的类别
    - 返回过滤后的 labels 和 emb，二者 index 完全一致

    ==========================
    参数说明 / Parameters
    ==========================

    :param emb: pd.DataFrame
        降维后的 2D/ND 嵌入矩阵，index 为 sample IDs。
        Embedding matrix (2D/ND), index must match sample IDs.

    :param labels: pd.Series
        标签序列，index 与 emb 对齐。
        Label series aligned with embedding rows.

    :param collapse_y:  boolean defeat = False
        是否将 Y haplogroup 映射到大类
        If True, collapse Y haplogroups into larger haplogroup classes.

    :param invalid_values: list | None
        要过滤掉的标签值（黑名单）。
        Default：["", " ", "NA", "N/A", "na", "..",
                  "n/a(female)", "n/a(sex unknown)", "n/a(female)"]
        List of invalid labels to remove.

    :param whitelist: list | None
        若提供，仅保留该列表中的标签类别。
        If provided, keep only these classes.

    ==========================
    返回值 / Returns
    ==========================
    :return emb_clean, labels_clean:
        - emb_clean: 过滤后的 embedding
        - labels_clean: 过滤后的标签（与 emb_clean 对齐）
    """

    # ---- 默认无效类别（可扩展）----
    if invalid_values is None:
        invalid_values = {
            "", " ", ".", "..", "...",
            "NA", "N/A", "na", "Na", "nA", "n/a", "n\\a",
            "nan", "NaN",

            # ADNA 中常见性别型 n/a (female)
            "n/a (female)", "n/a(female)", "n/a(Female)",
            "n/a (Female)", "n/a  (sex unknown)",
            "n/a(sex unknown)", "n/a (sex unknown)",
            "n/a",

            # 其他“未知”标注
            "unknown", "Unknown", "UNKNOWN",
            "undetermined", "Undetermined", "UNDETERMINED",
            "-", "—", "_", "?"
        }

    # ===== 2. 全部转成 str，并 strip 空格 =====
    labels_clean = labels.astype(str).str.strip()

    # ===== 3. 全部转成小写，便于比较 =====
    normalized = labels_clean.str.lower()

    # 将 invalid 映射为统一小写
    invalid_lower = {v.lower() for v in invalid_values}

    # ===== 4. 标记有效标签 =====
    mask_valid = ~normalized.isin(invalid_lower)

    # ===== 5. 若有 whitelist，进一步过滤 =====
    if whitelist is not None:
        whitelist_lower = {w.lower() for w in whitelist}
        mask_valid &= normalized.isin(whitelist_lower)

    # ===== 6. 根据 valid index 过滤 embedding =====
    valid_idx = labels.index[mask_valid]

    emb_clean = emb.loc[valid_idx].reset_index(drop=True)
    labels_clean = labels_clean.loc[valid_idx].reset_index(drop=True)

    if collapse_y:
        labels_clean = _collapse_y_haplogroup(labels_clean)

    return emb_clean, labels_clean


def _collapse_y_haplogroup(y: pd.Series) -> pd.Series:
    """
    将 Y haplogroup 映射到大类（Macro Haplogroup）
    Collapse detailed Y haplogroups to macro-level (first capital letter).

    例如：
        "R1b1a2"   → "R"
        "R-M269"   → "R"
        "J2a1"     → "J"
        "E1b1a"    → "E"

    说明:
        - 此函数不会删除无效标签（如 n/a），clean_labels_and_align 会处理
        - 仅对有效的 Y 值进行“取首字母大写”
    """

    y = y.astype(str).str.strip()

    macro = []
    for v in y:
        v = v.strip()

        # 无效值（n/a, .., unknown）保持原样，由 clean() 去掉
        if len(v) == 0:
            macro.append(v)
            continue

        first = v[0].upper()

        # 只对 alphabet 开头的 haplogroup 做宏类合并
        if first.isalpha():
            macro.append(first)
        else:
            macro.append(v)

    return pd.Series(macro, index=y.index, name=y.name + "_macro")
