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
    按样本 ID 对齐基因型矩阵与注释表，仅保留两表共有的样本，并保证行顺序一致。

    :param ids: 样本 ID 序列（与 X 的行一一对应）。
    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param meta: 注释表 (pd.DataFrame)，包含样本 ID 以及标签信息（如 Y haplogroup）。
    :param id_col: 注释表中的样本 ID 列名（默认 "Genetic ID"）。
    :return: (X_aligned, meta_aligned)
             - X_aligned: 仅保留共有样本后的基因型矩阵（重新从 0 开始索引）。
             - meta_aligned: 与 X_aligned 行顺序一致的注释表。
    注意:
        - 若注释表中缺失某些样本，将被自动丢弃；仅保留交集。
    """
    print("[INFO] Align by ID:")
    ids_unique = ids.drop_duplicates().reset_index(drop=True)
    meta_ids = meta[id_col].drop_duplicates()
    common_ids = ids_unique[ids_unique.isin(meta_ids)]

    mask = ids.isin(common_ids)
    X_aligned = X.loc[mask].reset_index(drop=True)
    meta_aligned = meta.set_index(id_col).loc[common_ids].reset_index()
    print(f"[OK] 共 {len(common_ids)} 个样本在两表中匹配 ({len(ids)}→{len(common_ids)})")
    return X_aligned, meta_aligned


def compute_missing_rates(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    计算缺失率（样本维度 & SNP 维度）。
        - 0 = 参考等位基因
        - 1 = 变异等位基因
        - 3 = 缺失

    :param X: 基因型矩阵 (pd.DataFrame)。
    :return: (sample_missing, snp_missing)
             - sample_missing: 每个样本（行）的缺失率 (0~1)。
             - snp_missing: 每个 SNP（列）的缺失率 (0~1)。
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


def filter_by_missing(X: pd.DataFrame, sample_missing: pd.Series, snp_missing: pd.Series, max_sample_missing: float = 0.8, max_snp_missing: float = 0.8) -> pd.DataFrame:
    """
    按缺失率阈值过滤样本与 SNP（默认阈值较宽松，可在后续调整）。

    :param X: 基因型矩阵。
    :param sample_missing: 每个样本的缺失率 (pd.Series)。
    :param snp_missing: 每个 SNP 的缺失率 (pd.Series)。
    :param max_sample_missing: 样本级最大缺失率阈值（默认 0.8）。
    :param max_snp_missing: SNP 级最大缺失率阈值（默认 0.8）。
    :return: 过滤后的矩阵 (pd.DataFrame)，索引从 0 重新开始。
    """
    print("[INFO] Filter by missing rate:")
    keep_rows = sample_missing <= max_sample_missing
    keep_cols = snp_missing[snp_missing <= max_snp_missing].index
    X_filtered = X.loc[keep_rows, keep_cols].reset_index(drop=True)
    print(f"[INFO] 样本: {keep_rows.sum()}/{len(X)} 保留, SNP: {len(keep_cols)}/{X.shape[1]} 保留")

    if X_filtered.empty:
        raise ValueError("[ERROR] 过滤后矩阵为空，请调整阈值。")

    return X_filtered


def _fill_mode(Z: pd.DataFrame, fallback: float = 1, save_disk: bool = True) -> pd.DataFrame:
    """
    列众数填补（大规模优化版）
    ===================================================
    - 小规模数据：直接在内存中执行；
    - 大规模数据：使用 NumPy memmap + PyArrow Parquet 边计算边落盘；
    - 自动路径管理：中间文件保存在 data/processed，最终结果移动至 data/results。

    :param Z: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :param fallback: float
        若整列均为空，则使用该值。
    :param save_disk: bool, default=True
        是否在超大规模数据时启用磁盘落盘模式。
    :return: pd.DataFrame
        填补后的矩阵；若启用落盘模式则返回空 DataFrame（结果已保存为 Parquet）。
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
        part_path = shard_dir / f"part_{part_id:03d}.parquet"
        table = pa.Table.from_pandas(block_filled.astype("float32"), preserve_index=False)
        pq.write_table(table, part_path, compression="zstd")

        # === 记录列信息到索引 ===
        col_index_meta.append({
            "part": f"part_{part_id:03d}.parquet",
            "columns": block.columns.tolist()
        })

    # === 保存列索引元数据 ===
    meta_path = shard_dir / "columns_index.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(col_index_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Sharded parquet written ({num_parts} files).")
    print(f"[OK] Saved in processed directory → {shard_dir}")
    print(f"[HINT] To load: ds.dataset(r'{shard_dir}', format='parquet')")

    return pd.DataFrame()


def _fill_mean(X: pd.DataFrame) -> pd.DataFrame:
    """
    列均值填补（矢量化实现）。
    ===================================================
    使用每列均值替代缺失值，适合连续型数据。

    :param X: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :return: pd.DataFrame
        填补后的矩阵。

    说明:
        - 对每列单独计算均值；
        - 用 fillna() 替换 NaN；
        - 输出与原矩阵列名一致。
    """
    # pandas fillna 本身矢量化，不会触发碎片警告
    result = X.fillna(X.mean())
    result.columns = X.columns
    result.index = X.index
    return result


def _fill_knn(X: pd.DataFrame, n_neighbors: int = 5, metric: str = "nan_euclidean") -> pd.DataFrame:
    """
    基于 sklearn 的 KNN 填补（Euclidean）
    ===================================================
    使用 KNNImputer，计算样本间欧氏距离（忽略 NaN）。

    :param X: pd.DataFrame
        基因型矩阵。
    :param n_neighbors: int, default=5
        近邻数量。
    :param metric: str, default="nan_euclidean"
        距离度量方式。
    :return: pd.DataFrame
        填补后的矩阵。

    说明:
        - 使用 sklearn.impute.KNNImputer；
        - 自动忽略 NaN 计算距离；
        - 大型数据集会自动提示性能警告。
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
    ===================================================
    使用未归一化汉明距离的 KNN 填补方法。（性能较慢，仅推荐小样本使用）

    :param X: pd.DataFrame
        基因型矩阵（值 ∈ {0,1,NaN}）。
    :param n_neighbors: int, default=5
        近邻数量。
    :param fallback: float, default=1.0
        若所有邻居均缺失时的回退值。
    :param strategy: str, default="mode"
        填补策略："mode" / "round" / "mean" / "prob"。
    :param random_state: int | None
        随机种子，仅在 "prob" 下生效。
    :return: pd.DataFrame
        填补后的矩阵。

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
    Adaptive Weighted Hamming KNN（自适应加权填补）
    ===================================================
    采用自适应衰减 α 的加权 KNN 填补算法。
    支持向量化与并行，适合高维稀疏 SNP 矩阵。

    :param X: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :param n_neighbors: int, default=5
        近邻数量。
    :param fallback: float, default=1.0
        回退值。
    :param target_decay: float, default=0.5
        控制权重中位邻居衰减比例。
    :param strategy: str, default="mode"
        填补策略："mode"/"round"/"mean"/"prob"。
    :param log_samples: int, default=3
        日志样本数量（打印 α 与部分权重）。
    :param random_state: int | None
        随机种子。
    :param parallel: bool, default=True
        是否启用列级并行。
    :return: pd.DataFrame
        填补后的矩阵。

    特性说明:
        - 向量化 α 与权重计算；
        - 列级并行填补；
        - 可打印部分样本的权重分布；
        - 高精度、速度快，适合中大规模数据。
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
    BallTree 加速 Hamming KNN 填补
    ===================================================
    基于 BallTree 的 Hamming 距离加速实现，
    支持列向量化与多核并行。

    :param X: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :param n_neighbors: int, default=5
        近邻数量。
    :param fallback: float, default=1.0
        若所有邻居均缺失时的回退值。
    :param strategy: str, default="mode"
        填补策略："mode"/"mean"/"round"。
    :param parallel: bool, default=False
        是否启用并行。
    :param n_jobs: int, default=-1
        并行任务数（默认所有 CPU 核）。
    :return: pd.DataFrame
        填补后的矩阵。

    特性说明:
        - BallTree 空间索引加速；
        - 向量化 + 列级并行；
        - 速度快、内存占用低；
        - 推荐中等规模数据使用。
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


def _fill_knn_faiss(
    X: pd.DataFrame,
    n_neighbors: int = 5,
    n_components: int = 50,
    fallback: float = 1.0,
    strategy: str = "mode",
    random_state: int | None = 42,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """
    PCA + Faiss 近似 KNN 填补（增强版，支持 IncrementalPCA）
    ==============================================================
    适用于大规模基因型矩阵（>10k 样本），
    在 PCA 阶段自动选择普通 PCA 或 IncrementalPCA，
    并使用 Faiss 加速最近邻搜索（CPU/GPU 均可）。

    :param X: pd.DataFrame
        基因型矩阵 (rows = samples, cols = SNPs; values ∈ {0, 1, NaN})
    :param n_neighbors: int
        KNN 近邻数量。
    :param n_components: int
        降维后的维度数。
    :param fallback: float
        当所有邻居均缺失时使用的回退值。
    :param strategy: str
        填补策略："mode" | "mean" | "round"。
    :param random_state: int | None
        随机种子。
    :param batch_size: int
        IncrementalPCA 的每批样本数。
    :return: pd.DataFrame
        填补后的 DataFrame。
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
    自动选择最优 KNN 填补算法（Auto-select）
    ===================================================
    根据样本规模与缺失率自动选择最佳策略。

    策略逻辑：
        1. 内存估算 > 20 GB → 众数填补 (mode)
        2. 缺失率 ≥ 40% → Adaptive Hamming (prob)
        3. n < 500 → Hamming(abs)
        4. 500 ≤ n ≤ 5000 → BallTree 并行 KNN
        5. 5000 < n ≤ 10000 → Adaptive Weighted Hamming
        6. n > 10000 → PCA + Faiss (自动切换 IncrementalPCA)

    :param X: pd.DataFrame
        基因型矩阵。
    :param n_neighbors: int
        近邻数量。
    :param fallback: float
        回退值。
    :param missing_threshold: float
        缺失率阈值（默认 0.4）。
    :param random_state: int | None
        随机种子。
    :param verbose: bool
        是否打印日志。

    :return: pd.DataFrame
        填补后的矩阵。
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
    ===================================================
    根据参数选择填补方法。

    支持方法：
        - "mode"                  → 众数填补；
        - "mean"                  → 均值填补；
        - "knn"                   → sklearn KNN；
        - "knn_hamming_abs"       → 等权 Hamming；
        - "knn_hamming_adaptive"  → 自适应加权 Hamming；
        - "knn_hamming_balltree"  → BallTree 加速；
        - "knn_hamming_faiss"     → PCA + Faiss 近似 KNN 填补
        - "knn_auto"              → 自动选择策略。

    :param X: pd.DataFrame
        基因型矩阵（0=参考等位基因, 1=变异等位基因, 3=缺失）。
    :param method: str
        填补方法。
    :param n_neighbors: int, default=5
        KNN 近邻数。
    :return: pd.DataFrame
        填补后的矩阵。
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
        raise ValueError(f"未知填补方法: {method}")

    after_nans = filled.isna().sum().sum()
    replaced = before_nans - after_nans

    if after_nans > 0:
        print(f"[WARN] 填补后仍存在 {after_nans} 个 NaN（可能为异常列）。")
    print(f"[OK] {method.upper():<5} 填补完成 | 替换缺失值数: {replaced} | 残留 NaN: {after_nans}")

    return filled


def grouped_imputation(X, labels, method: str = "mode"):
    """
    按外部标签分组执行缺失值填补（封装版）
    ===================================================
    :param X: pd.DataFrame
        原始基因型矩阵。
    :param labels: pd.Series or None
        外部标签列（例如 World Zone / Y haplogroup）。
        若为 None，则执行全局填补。
    :param method: str
        填补方法，如 "mode"、"knn_hamming_adaptive"。
    :return: pd.DataFrame
        分组填补后的矩阵。
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