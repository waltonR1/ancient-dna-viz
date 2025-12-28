"""
preprocess.py
=============

数据预处理模块（Preprocessing Layer）
Preprocessing module for genotype matrix cleaning, filtering, and imputation.

本模块是基因型数据分析管线的核心组成部分，
负责对原始基因型矩阵执行缺失值检测、样本对齐、
缺失率计算、过滤以及多策略缺失值填补。

This module forms the preprocessing layer of the genotype data pipeline.
It handles missing-value detection, sample alignment, missing-rate analysis,
filtering, and imputation using multiple optimized strategies.

Functions
---------
align_by_id
    按样本 ID 对齐基因型矩阵与样本注释表。
    Align genotype matrix with metadata table by sample IDs.

compute_missing_rates
    计算样本与 SNP 维度的缺失率。
    Compute missing rates per sample and per SNP.

filter_by_missing
    按缺失率阈值过滤样本与 SNP。
    Filter samples and SNPs based on missing-rate thresholds.

filter_meta_by_rows
    根据样本保留掩码同步过滤元数据表。
    Filter metadata table using a row-selection mask.

impute_missing
    通用缺失值填补入口函数。
    Unified entry point for missing-value imputation.

grouped_imputation
    基于外部分组标签的分组填补。
    Perform group-wise imputation based on external labels.

_fill_mode, _fill_mean, _fill_knn, ...
    各类具体缺失值填补算法的内部实现。
    Internal implementations of various imputation strategies.

clean_labels_and_align
    清洗分类标签并与 embedding 对齐。
    Clean categorical labels and align them with embedding results.

Description
-----------
本模块专注于解决基因型矩阵在真实数据场景中
普遍存在的缺失、不一致和规模受限问题，
并提供从简单统计方法到高级 KNN / Faiss 的
多层级、可扩展缺失值填补方案。

The module addresses common issues in real-world genotype datasets,
including missing values, inconsistencies, and scalability constraints.
It provides a hierarchical set of imputation strategies,
ranging from simple statistical methods to advanced KNN and Faiss-based approaches.

模块输出的数据结构被直接用于后续的
降维（embedding）、聚类（clustering）以及可视化分析流程。

The outputs of this module are directly consumed by downstream
embedding, clustering, and visualization components.
"""

import json
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import multiprocessing
import warnings
from tqdm import tqdm
import psutil, time
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def align_by_id(
        ids: pd.Series,
        X: pd.DataFrame,
        meta: pd.DataFrame,
        id_col: str = "Genetic ID"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按样本 ID 对齐基因型矩阵与注释表

    Align genotype matrix and metadata table by sample IDs.

    本函数以 ``ids`` 与注释表中的 ``id_col`` 为依据，
    计算二者的样本 ID 交集，并据此筛选与重排基因型矩阵
    与注释表，仅保留共有样本。

    The function computes the intersection between the sample IDs
    provided in ``ids`` and those in the metadata column ``id_col``,
    and filters both the genotype matrix and metadata table
    to keep only shared samples.

    对齐规则（严格对应代码实现）：
    - 基因型矩阵 ``X`` 的样本顺序 **保持原顺序不变**
    - 注释表 ``meta`` 的样本顺序 **按照交集 ID 顺序重排**
    - 两个输出在行数上应一致，并代表同一批样本

    Alignment rules (as implemented):
    - The genotype matrix ``X`` preserves its original row order
    - The metadata table ``meta`` is reordered according to
      the intersecting sample IDs
    - Both outputs correspond to the same set of samples

    Parameters
    ----------
    ids : pandas.Series
        样本 ID 序列，与 ``X`` 的行一一对应。

        Series of sample IDs aligned with the rows of ``X``.

    X : pandas.DataFrame
        基因型矩阵，行表示样本，列表示 SNP 特征。

        Genotype matrix with samples as rows and SNPs as columns.

    meta : pandas.DataFrame
        样本注释表，必须包含样本 ID 列 ``id_col``。

        Metadata table containing the sample ID column ``id_col``.

    id_col : str, default="Genetic ID"
        注释表中样本 ID 的列名。

        Column name of sample IDs in the metadata table.

    Returns
    -------
    X_aligned : pandas.DataFrame
        对齐后的基因型矩阵，仅保留与注释表共有的样本，
        并重置行索引（``reset_index(drop=True)``）。

        Aligned genotype matrix containing only intersecting samples,
        with row indices reset.

    meta_aligned : pandas.DataFrame
        与 ``X_aligned`` 行数一致的注释表，
        样本顺序由交集 ID 决定。

        Metadata table aligned to the same samples.

    Raises
    ------
    ValueError
        当基因型矩阵与注释表之间不存在任何共有样本 ID 时抛出。

        Raised when no overlapping sample IDs are found
        between the genotype matrix and metadata.

    Notes
    -----
    - 本函数会将 ``ids`` 与 ``meta[id_col]`` 强制转换为字符串，
      以避免类型不一致导致的匹配失败。

      Both ``ids`` and ``meta[id_col]`` are explicitly cast to string
      to prevent type-mismatch issues during alignment.

    - 函数会打印对齐过程中的信息与一致性检查日志，
      但不会在不一致时强制终止执行。

      Informational and consistency-check logs are printed,
      but mismatched row counts only trigger warnings.
    """

    print("[INFO] Aligning genotype matrix and metadata by sample ID...")

    # 统一类型，防止类型不匹配
    ids = ids.astype(str)
    meta_ids = meta[id_col].astype(str)

    # 计算交集
    common_ids = pd.Index(ids).intersection(meta_ids)
    if len(common_ids) == 0:
        raise ValueError("[ERROR] No overlapping sample IDs between X and metadata.")

    # 按交集保留样本并保持顺序
    X_aligned = X.loc[ids.isin(common_ids)].reset_index(drop=True)
    meta_aligned = meta.set_index(id_col).loc[common_ids].reset_index()

    print(f"[OK] Matched {len(common_ids)} samples (from {len(ids)} → {len(common_ids)}).")

    # 一致性检查
    if len(X_aligned) != len(meta_aligned):
        print(f"[WARN] Row count mismatch after alignment: X={len(X_aligned)} vs meta={len(meta_aligned)}")
    else:
        print("[CHECK] Alignment successful. Row counts match exactly.")

    return X_aligned, meta_aligned


def compute_missing_rates(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    计算样本维度与 SNP 维度的缺失率

    Compute missing rates across sample and SNP dimensions.

    本函数基于基因型矩阵中的编码规则，
    将数值 ``3`` 视为缺失值，并在转换为 ``NaN`` 后，
    分别计算每个样本（行）与每个 SNP（列）的缺失比例。

    This function treats the value ``3`` in the genotype matrix
    as a missing value, converts it to ``NaN``, and computes
    missing rates per sample (row-wise) and per SNP (column-wise).

    编码规则（严格对应实现）：
    - ``0`` : 参考等位基因（reference allele）
    - ``1`` : 变异等位基因（alternate allele）
    - ``3`` : 缺失值（missing value）

    Encoding scheme (as implemented):
    - ``0`` : reference allele
    - ``1`` : alternate allele
    - ``3`` : missing value

    Parameters
    ----------
    X : pandas.DataFrame
        基因型矩阵，行表示样本，列表示 SNP，
        其中缺失值使用整数编码 ``3`` 表示。

        Genotype matrix with samples as rows and SNPs as columns.
        Missing values must be encoded as ``3``.

    Returns
    -------
    sample_missing : pandas.Series
        每个样本的缺失率（取值范围 0–1），
        索引与 ``X`` 的行索引一致。

        Missing rate per sample (row-wise), ranging from 0 to 1.

    snp_missing : pandas.Series
        每个 SNP 的缺失率（取值范围 0–1），
        索引与 ``X`` 的列名一致。

        Missing rate per SNP (column-wise), ranging from 0 to 1.

    Notes
    -----
    - 函数会将 ``X`` 转换为 ``float32`` 的副本进行计算，
      原始输入矩阵不会被原地修改。

      The input matrix is copied and cast to ``float32``;
      the original DataFrame is not modified in place.

    - 为避免一次性生成巨大布尔矩阵，
      缺失值替换过程按列分块执行（默认每块 2000 列）。

      To reduce memory usage, missing-value replacement
      is performed in column-wise blocks (default size: 2000).

    - 缺失率计算基于 ``NaN`` 比例，
      使用 ``DataFrame.isna().mean`` 实现。

      Missing rates are computed as the proportion of ``NaN``
      values using ``isna().mean``.

    - 函数会打印处理进度与状态信息，但不影响返回结果。

      Informational messages are printed during execution,
      but they do not affect the returned values.
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


def filter_by_missing(
        X: pd.DataFrame,
        sample_missing: pd.Series,
        snp_missing: pd.Series,
        max_sample_missing: float = 0.8,
        max_snp_missing: float = 0.8
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    按缺失率阈值过滤样本与 SNP

    Filter samples and SNPs based on missing-rate thresholds.

    本函数根据样本级与 SNP 级缺失率，
    构造布尔掩码并同时过滤基因型矩阵的行（样本）
    与列（SNP），返回过滤后的矩阵以及样本保留掩码。

    This function filters both rows (samples) and columns (SNPs)
    of the genotype matrix based on sample-level and SNP-level
    missing-rate thresholds, and returns the filtered matrix
    together with the sample-selection mask.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型矩阵，行表示样本，列表示 SNP。

        Genotype matrix with samples as rows and SNPs as columns.

    sample_missing : pandas.Series
        每个样本的缺失率，索引与 ``X`` 的行一一对应。

        Missing rate per sample, indexed to match rows of ``X``.

    snp_missing : pandas.Series
        每个 SNP 的缺失率，索引与 ``X`` 的列名一致。

        Missing rate per SNP, indexed by SNP names.

    max_sample_missing : float, default=0.8
        样本允许的最大缺失率阈值。

        Maximum allowed missing rate per sample.

    max_snp_missing : float, default=0.8
        SNP 允许的最大缺失率阈值。

        Maximum allowed missing rate per SNP.

    Returns
    -------
    X_filtered : pandas.DataFrame
        过滤后的基因型矩阵，仅保留满足缺失率阈值的
        样本与 SNP，行索引已重置。

        Filtered genotype matrix containing only samples and
        SNPs passing the thresholds, with row indices reset.

    keep_rows : pandas.Series
        样本级布尔掩码，表示哪些样本被保留。

        Boolean mask indicating which samples are retained.

    Raises
    ------
    ValueError
        当过滤后矩阵为空时抛出，
        提示用户调整缺失率阈值。

        Raised when the filtered matrix is empty, indicating
        that the thresholds may be too strict.

    Notes
    -----
    - 样本筛选规则为：
      ``sample_missing <= max_sample_missing``。

      Samples are kept if
      ``sample_missing <= max_sample_missing``.

    - SNP 筛选规则为：
      ``snp_missing <= max_snp_missing``。

      SNPs are kept if
      ``snp_missing <= max_snp_missing``.

    - 函数会打印样本与 SNP 的保留数量信息，
      但不会影响返回结果。

      Informational messages about retained samples and SNPs
      are printed but do not affect the returned values.
    """

    print("[INFO] Filter by missing rate:")
    keep_rows: pd.Series = (sample_missing <= max_sample_missing).astype(bool)
    keep_cols = snp_missing[snp_missing <= max_snp_missing].index
    X_filtered = X.loc[keep_rows, keep_cols].reset_index(drop=True)
    print(f"[INFO] Sample: {keep_rows.sum()}/{len(X)} reserved, SNP: {len(keep_cols)}/{X.shape[1]} reserved")

    if X_filtered.empty:
        raise ValueError("[ERROR] The filtered matrix is empty. Please adjust the threshold.")

    return X_filtered, keep_rows


def filter_meta_by_rows(
        meta: pd.DataFrame,
        keep_rows: pd.Series,
        label: str = "meta"
) -> pd.DataFrame:
    """
    根据样本保留掩码过滤元数据表（带日志输出）

    Filter metadata table using a row-selection mask with logging.

    本函数根据布尔掩码 ``keep_rows`` 对元数据表 ``meta`` 执行行级过滤，
    并在过滤前后打印详细的统计日志，包括总行数、保留行数与移除比例。

    This function filters the metadata table ``meta`` using a boolean
    mask ``keep_rows`` and prints detailed logging information before
    and after filtering.

    Parameters
    ----------
    meta : pandas.DataFrame
        元数据表，行数应与生成 ``keep_rows`` 的基因型矩阵一致。

        Metadata table whose number of rows should match the source
        matrix used to generate ``keep_rows``.

    keep_rows : pandas.Series of bool
        样本级布尔掩码，通常来自 ``filter_by_missing``，
        ``True`` 表示对应样本被保留。

        Boolean mask indicating which rows to retain, typically
        produced by ``filter_by_missing``.

    label : str, default="meta"
        日志输出中使用的名称标识，
        仅用于打印信息，不影响过滤逻辑。

        Label name used only for logging output and diagnostics.

    Returns
    -------
    pandas.DataFrame
        过滤后的元数据表，行索引已重置。

        Filtered metadata table with reset row indices.

    Raises
    ------
    ValueError
        当过滤后的行数与 ``keep_rows`` 中 ``True`` 的数量不一致时抛出，
        用于一致性检查。

        Raised if the number of filtered rows does not match the
        number of ``True`` values in ``keep_rows``.

    Notes
    -----
    - 本函数假设 ``keep_rows`` 与 ``meta`` 在行顺序上是一一对应的。

      The function assumes that ``keep_rows`` is positionally aligned
      with the rows of ``meta``.

    - 参数 ``label`` 仅用于日志显示，不参与任何数据处理逻辑。

      The ``label`` parameter is used exclusively for logging and
      does not affect filtering behavior.
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


def _fill_mode(
        X: pd.DataFrame,
        fallback: float = 1,
        save_disk: bool = True
) -> pd.DataFrame:
    """
    基于列众数的缺失值填补（支持内存检测与分片落盘）

    Column-wise mode imputation with adaptive memory handling and optional disk output.

    本函数对基因型矩阵 ``X`` 按列执行众数（mode）填补，
    并根据数据规模与可用内存情况，自动选择：
    - 直接在内存中完成填补；
    - 或按列分片处理，并将结果写入 Parquet 文件以避免内存溢出。

    This function performs column-wise mode imputation on the
    genotype matrix ``X`` and automatically chooses between
    in-memory processing or sharded, on-disk output depending
    on data size and available system memory.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型矩阵，行表示样本，列表示 SNP。
        缺失值应已表示为 ``NaN``。

        Genotype matrix with samples as rows and SNPs as columns.
        Missing values must already be encoded as ``NaN``.

    fallback : float, default=1
        当某一列全部为缺失值（全 ``NaN``）时，
        用于填补该列的回退值。

        Fallback value used when an entire column contains only
        missing values.

    save_disk : bool, default=True
        是否在检测到内存不足或数据规模过大时，
        启用分片写入 Parquet 文件的磁盘模式。

        Whether to enable on-disk (Parquet) output when memory
        is insufficient or the dataset is too large.

    Returns
    -------
    pandas.DataFrame
        - 若在内存中完成填补：返回填补后的 DataFrame；
        - 若启用磁盘写入模式：返回一个空的 DataFrame，
          实际结果已写入 Parquet 文件。

        Returns the imputed DataFrame if processed in memory.
        Returns an empty DataFrame if on-disk mode is used
        (results are written to Parquet files).

    Notes
    -----
    - 众数计算基于 ``value_counts().idxmax()``，
      并忽略缺失值（``NaN``）。

      The mode of each column is computed using
      ``value_counts().idxmax()``, ignoring ``NaN`` values.

    - 当列中不存在任何非缺失值时，
      使用 ``fallback`` 作为填充值。

      If a column contains no non-missing values,
      the ``fallback`` value is used.

    - 当启用磁盘模式时，列将按批次处理，
      每个批次的结果写入独立的 Parquet 分片文件。

      In on-disk mode, columns are processed in batches,
      and each batch is written to a separate Parquet shard.

    - 本函数包含内存使用情况的检测与日志输出，
      但不影响最终填补逻辑。

      The function performs memory checks and prints
      diagnostic logs, which do not affect the imputation logic.
    """

    n_rows, n_cols = X.shape
    est_size_gb = n_rows * n_cols * 4 / 1024**3
    free_mem_gb = psutil.virtual_memory().available / 1024**3

    print(f"[INFO] Mode filling started: {n_rows}×{n_cols} | est={est_size_gb:.1f} GB | free_mem={free_mem_gb:.1f} GB")

    # 路径准备
    root_dir = Path(__file__).resolve().parents[2]  # 项目根目录
    processed_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "data" / "results"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    shard_dir = processed_dir / f"mode_filled_{timestamp}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # 判定是否使用磁盘模式
    use_disk = save_disk and (est_size_gb > free_mem_gb * 0.6 or est_size_gb > 8)
    # 小规模数据：在内存中直接填补
    if not use_disk:
        print(f"[INFO] Using in-memory mode filling (dataset size={est_size_gb:.1f} GB).")

        filled_cols = {}  # 用字典暂存所有填补结果
        for col in tqdm(X.columns, desc="Mode filling", ncols=100):
            col_data = X[col]
            if col_data.isna().all():
                mode_val = fallback
            else:
                mode_val = col_data.mode(dropna=True).iloc[0]
            filled_cols[col] = col_data.fillna(mode_val)

        # 一次性拼接，避免碎片化
        result = pd.concat(filled_cols, axis=1)
        result.columns = X.columns
        result.index = X.index
        print(f"[OK] In-memory mode filling complete.")
        return result

    # 大规模数据，启用磁盘写入（列分片方案）
    print(f"[AUTO] Large dataset detected — using sharded Parquet write mode → {shard_dir}")
    batch_cols = 2000
    num_parts = (n_cols + batch_cols - 1) // batch_cols
    # 创建索引列表
    col_index_meta = []

    for part_id, start in enumerate(tqdm(range(0, n_cols, batch_cols), desc="Mode filling (sharded)", ncols=100)):
        end = min(start + batch_cols, n_cols)
        block = X.iloc[:, start:end]

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

    #     # 记录列信息到索引
        col_index_meta.append({
            "part": f"part_{part_id:03d}.parquet",
            "columns": block.columns.tolist()
        })

    # 保存列索引元数据
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

    本函数对每一列独立计算均值，
    并使用该均值替换列中的缺失值（NaN），
    采用 Pandas 内部的矢量化操作实现。

    This function computes the mean of each column and replaces
    missing values (NaN) with the corresponding column mean,
    using Pandas' vectorized operations.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型或数值特征矩阵，行表示样本，列表示特征。
        缺失值需以 NaN 表示。

        Numeric feature matrix with samples as rows and features
        as columns. Missing values must be represented as NaN.

    Returns
    -------
    pandas.DataFrame
        填补后的矩阵，行索引与列名与输入完全一致。

        Imputed DataFrame with the same shape, index, and columns
        as the input.

    Notes
    -----
    - 均值通过 ``X.mean()`` 计算，默认忽略 NaN 值。

      Column means are computed using ``X.mean()``,
      which ignores NaN values by default.

    - 该实现完全基于 Pandas 的矢量化操作，
      不包含显式循环，不会触发碎片化（fragmentation）警告。

      This implementation relies entirely on Pandas vectorized
      operations, avoiding explicit loops and fragmentation warnings.

    - 适用于连续数值型特征；
      对于离散型或类别型 SNP 编码数据，均值填补可能不具备生物学意义。

      Suitable for continuous numeric features; mean imputation
      may be inappropriate for categorical or discrete SNP encodings.
    """

    # pandas fillna 本身矢量化，不会触发碎片警告
    result = X.fillna(X.mean())
    result.columns = X.columns
    result.index = X.index
    return result


def _fill_knn(
        X: pd.DataFrame,
        n_neighbors: int = 5,
        metric: str = "nan_euclidean"
) -> pd.DataFrame:
    """
    基于 sklearn 的 KNN 缺失值填补

    KNN-based missing value imputation using sklearn.

    本函数使用 ``sklearn.impute.KNNImputer`` 对基因型矩阵
    按样本间距离执行 KNN 缺失值填补。
    距离计算基于指定的 ``metric``，并在计算过程中自动忽略 NaN 值。

    This function performs KNN-based missing value imputation
    using ``sklearn.impute.KNNImputer``. Pairwise distances
    between samples are computed using the specified ``metric``,
    while NaN values are ignored during distance computation.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型或数值特征矩阵，行表示样本，列表示特征。
        缺失值需以 NaN 表示。

        Numeric feature matrix with samples as rows and features
        as columns. Missing values must be represented as NaN.

    n_neighbors : int, default=5
        用于 KNN 填补的近邻数量（k 值）。

        Number of nearest neighbors used for KNN imputation.

    metric : str, default="nan_euclidean"
        距离度量方式，默认使用忽略 NaN 的欧氏距离。

        Distance metric used for computing sample-to-sample
        distances. The default ``"nan_euclidean"`` ignores NaN values.

    Returns
    -------
    pandas.DataFrame
        填补后的矩阵，行索引与列名与输入完全一致。

        Imputed DataFrame with the same index and columns
        as the input.

    Notes
    -----
    - 本函数直接调用 ``KNNImputer.fit_transform``，
      不对数据进行标准化或缩放。

      The function directly calls ``KNNImputer.fit_transform``
      and does not perform any feature scaling or normalization.

    - 当样本数量超过 5000 时，会打印性能警告，
      提示 KNN 填补可能存在较高计算开销。

      A performance warning is printed when the number of
      samples exceeds 5000, as KNN imputation may be slow.

    - 填补结果通过 ``pd.DataFrame`` 封装返回，
      并显式保留原始行索引与列名。

      The imputed result is wrapped in a ``pandas.DataFrame``
      with original indices and column names preserved.
    """

    if X.shape[0] > 5000:
        print(f"[WARN] Large dataset detected ({X.shape[0]} samples) — KNN imputation may be slow.")

    imputer = KNNImputer(n_neighbors=n_neighbors, metric=metric)
    M = imputer.fit_transform(X)
    print(f"[OK] sklearn.KNNImputer complete — metric={metric}, k={n_neighbors}")
    return pd.DataFrame(M, columns=X.columns, index=X.index)


def _fill_knn_hamming_abs(
        X: pd.DataFrame,
        n_neighbors: int = 5,
        fallback: float = 1.0,
        strategy: str = "mode",
        random_state: int | None = None
) -> pd.DataFrame:
    """
    基于未归一化汉明距离的等权 KNN 填补（逐元素循环实现）

    Equal-weight KNN imputation using unnormalized Hamming distance (loop-based).

    本函数使用 ``sklearn.neighbors.NearestNeighbors(metric="hamming")``
    在样本之间构建近邻关系，并将返回的归一化汉明距离乘以特征数，
    得到“未归一化”的汉明距离（Hamming(abs)）。
    然后对矩阵中的每个 NaN 位置，按指定策略从近邻样本的同列取值中
    聚合或采样得到填补值。

    This function builds a nearest-neighbor graph using
    ``NearestNeighbors(metric="hamming")``. The normalized Hamming
    distances returned by sklearn are multiplied by the number of
    features to obtain unnormalized distances (Hamming(abs)).
    For each NaN entry, the value is imputed from neighbors in the
    same column according to the chosen strategy.

    Parameters
    ----------
    X : pandas.DataFrame
        输入矩阵（函数内部会 ``copy``），缺失值以 NaN 表示。

        Input matrix (copied internally). Missing values must be NaN.

    n_neighbors : int, default=5
        近邻数量（实际建模时会使用 ``n_neighbors + 1``，包含自身后再剔除）。

        Number of neighbors (the model uses ``n_neighbors + 1`` to
        include self, which is then removed).

    fallback : float, default=1.0
        当某个缺失位置在所有近邻样本中该列也均为缺失时，
        使用的回退填充值。

        Fallback value used when all neighbor values in the same
        column are missing.

    strategy : str, default="mode"
        缺失值的填补策略，可选：
        ``"mode"``, ``"round"``, ``"mean"``, ``"prob"``。

        Imputation strategy. Supported values are:
        ``"mode"``, ``"round"``, ``"mean"``, ``"prob"``.

    random_state : int or None, default=None
        随机种子，仅在 ``strategy="prob"`` 时用于二项采样。

        Random seed used only when ``strategy="prob"`` for
        Bernoulli sampling.

    Returns
    -------
    pandas.DataFrame
        填补后的矩阵，行索引与列名与输入保持一致。

        Imputed DataFrame with the same index and columns as input.

    Raises
    ------
    ValueError
        当 ``strategy`` 不是支持的取值之一时抛出。

        Raised when ``strategy`` is not one of the supported values.

    Notes
    -----
    - sklearn 的 Hamming 距离无法直接处理 NaN，
      因此本实现先将 NaN 替换为哑值 ``-1`` 用于近邻搜索。

      sklearn's Hamming metric cannot handle NaN directly, so NaNs
      are replaced with a dummy value ``-1`` for neighbor search.

    - 近邻搜索结果包含样本自身，因此会剔除自身：
      ``indices = indices[:, 1:]``。

      The neighbor list includes the sample itself, which is removed
      via ``indices = indices[:, 1:]``.

    - 填补过程为双重循环（样本 × 特征），并使用 ``tqdm`` 显示进度，
      因此在大规模数据上可能较慢。

      The imputation uses nested loops over samples and features and
      shows progress with ``tqdm``, so it may be slow on large datasets.

    - ``"prob"`` 策略使用近邻均值作为概率，
      通过 ``rng.binomial(1, p)`` 进行二项采样。

      The ``"prob"`` strategy uses the neighbor mean as probability
      and samples via ``rng.binomial(1, p)``.
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


def _fill_knn_hamming_adaptive(
        X: pd.DataFrame,
        n_neighbors: int = 5,
        fallback: float = 1.0,
        target_decay: float = 0.5,
        strategy: str = "mode",
        log_samples: int = 3,
        random_state: int | None = None,
        parallel: bool = True
) -> pd.DataFrame:
    """
    基于自适应指数加权的汉明距离 KNN 填补

    Adaptive weighted KNN imputation using unnormalized Hamming distance.

    本函数基于样本间的汉明距离构建近邻关系，
    并根据每个样本的邻居距离分布自适应计算指数衰减系数，
    对邻居样本在同一列上的取值进行加权聚合，
    从而完成缺失值填补。

    This function builds a nearest-neighbor graph using Hamming
    distance between samples and adaptively derives an exponential
    decay coefficient per sample based on its neighbor distance
    distribution. Missing values are imputed by aggregating
    neighbor values in the same column using distance-based weights.

    Parameters
    ----------
    X : pandas.DataFrame
        输入矩阵（函数内部会 ``copy``），缺失值以 NaN 表示。

        Input matrix (copied internally). Missing values must be NaN.

    n_neighbors : int, default=5
        近邻数量（实际搜索时使用 ``n_neighbors + 1``，
        并在结果中剔除样本自身）。

        Number of neighbors (``n_neighbors + 1`` are queried and
        the self-neighbor is removed).

    fallback : float, default=1.0
        当某个缺失位置在所有邻居中该列均为缺失时，
        使用的回退填充值。

        Fallback value used when all neighbor values are missing.

    target_decay : float, default=0.5
        用于确定指数衰减系数的目标比例。
        该值表示“中位距离邻居”的相对权重。

        Target decay ratio used to derive the exponential
        decay coefficient from the median neighbor distance.

    strategy : str, default="mode"
        缺失值聚合策略，可选：
        ``"mode"``, ``"round"``, ``"mean"``, ``"prob"``。

        Strategy used to aggregate neighbor values:
        ``"mode"``, ``"round"``, ``"mean"``, ``"prob"``.

    log_samples : int, default=3
        随机选取用于日志输出的样本数量，
        打印其衰减系数、距离统计及前若干权重。

        Number of randomly selected samples for logging
        decay coefficients and neighbor weights.

    random_state : int or None, default=None
        随机种子，仅在 ``strategy="prob"`` 时
        用于二项采样的可重复性。

        Random seed used only for the ``"prob"`` strategy.

    parallel : bool, default=True
        是否在列维度上启用并行计算（Joblib）。
        仅当列数大于 500 时生效。

        Whether to enable column-level parallelism via Joblib.
        Only effective when the number of columns exceeds 500.

    Returns
    -------
    pandas.DataFrame
        填补后的矩阵，列名与输入一致。

        Imputed DataFrame with the same columns as the input.

    Raises
    ------
    ValueError
        当 ``strategy`` 不是支持的取值之一时抛出。

        Raised when ``strategy`` is not supported.

    Notes
    -----
    - sklearn 的 Hamming 距离无法处理 NaN，
      因此本实现先将 NaN 替换为 ``-1`` 用于近邻搜索。

      sklearn's Hamming metric cannot handle NaN directly,
      so NaN values are replaced with ``-1`` before neighbor search.

    - ``BallTree(metric="hamming")`` 返回的是归一化汉明距离，
      本函数将其乘以特征数以恢复未归一化距离。

      The Hamming distances returned by ``BallTree`` are normalized;
      they are multiplied by the number of features to obtain
      unnormalized distances.

    - 衰减系数按样本级自适应计算：
      ``alpha = log(1 / target_decay) / median_distance``。

      The decay coefficient is computed per sample as:
      ``alpha = log(1 / target_decay) / median_distance``.

    - 填补逻辑在列级执行：
      对每个缺失位置，从对应样本的邻居中提取同列取值，
      并按所选策略进行加权聚合。

      Imputation is performed column-wise: for each missing entry,
      neighbor values in the same column are aggregated using
      the chosen strategy.

    - 实现包含 Python 循环与可选并行，
      在大规模数据上可能存在较高计算开销。

      The implementation involves Python-level loops with
      optional parallelism and may be computationally intensive
      on large datasets.
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


def _fill_knn_hamming_balltree(
        X: pd.DataFrame,
        n_neighbors: int = 5,
        fallback: float = 1.0,
        strategy: str = "mode",
        parallel: bool = False,
        n_jobs: int = -1
) -> pd.DataFrame:
    """
    基于 BallTree 的汉明距离 KNN 缺失值填补

    Hamming-distance KNN imputation using BallTree.

    本函数基于 ``sklearn.neighbors.BallTree(metric="hamming")``
    在样本之间构建近邻关系，并对缺失值位置从邻居样本的
    同一列取值中进行聚合，从而完成 KNN 填补。

    This function builds a nearest-neighbor graph using
    ``BallTree(metric="hamming")`` and imputes missing values
    by aggregating neighbor values in the same column.

    Parameters
    ----------
    X : pandas.DataFrame
        输入矩阵（函数内部会 ``copy``），缺失值以 NaN 表示。

        Input matrix (copied internally). Missing values must be NaN.

    n_neighbors : int, default=5
        近邻数量（实际搜索时使用 ``n_neighbors + 1``，
        并在结果中剔除样本自身）。

        Number of neighbors (``n_neighbors + 1`` are queried and
        the self-neighbor is removed).

    fallback : float, default=1.0
        当某个缺失位置在所有邻居中该列均为缺失时，
        使用的回退填充值。

        Fallback value used when all neighbor values are missing.

    strategy : str, default="mode"
        缺失值聚合策略，可选：
        ``"mode"``, ``"mean"``, ``"round"``。

        Strategy used to aggregate neighbor values:
        ``"mode"``, ``"mean"``, ``"round"``.

    parallel : bool, default=False
        是否在列维度上启用并行计算（Joblib）。

        Whether to enable column-level parallelism via Joblib.

    n_jobs : int, default=-1
        并行任务数，仅在 ``parallel=True`` 时生效。
        ``-1`` 表示使用所有可用 CPU 核心。

        Number of parallel jobs used when ``parallel=True``.
        ``-1`` means using all available CPU cores.

    Returns
    -------
    pandas.DataFrame
        填补后的矩阵，列名与输入一致。

        Imputed DataFrame with the same columns as the input.

    Raises
    ------
    ValueError
        当 ``strategy`` 不是支持的取值之一时抛出。

        Raised when ``strategy`` is not supported.

    Notes
    -----
    - sklearn 的 Hamming 距离无法直接处理 NaN，
      因此本实现会先将 NaN 替换为 ``-1`` 用于近邻搜索。

      sklearn's Hamming metric cannot handle NaN directly,
      so NaN values are replaced with ``-1`` before neighbor search.

    - ``BallTree(metric="hamming")`` 返回的是归一化汉明距离，
      本函数会将其乘以特征数以恢复未归一化距离。

      The Hamming distances returned by ``BallTree`` are normalized
      and multiplied by the number of features to obtain
      unnormalized distances.

    - 填补逻辑在列级执行：
      对每个缺失位置，从对应样本的邻居中提取同列取值，
      并按所选策略进行聚合。

      Imputation is performed column-wise: for each missing entry,
      neighbor values in the same column are aggregated using
      the chosen strategy.

    - 是否启用并行仅由 ``parallel`` 参数控制，
      本函数不会根据数据规模自动切换执行模式。

      Parallel execution is controlled solely by ``parallel``;
      no automatic switching based on dataset size is performed.
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


def impute_missing(
        X: pd.DataFrame,
        method: str = "mode",
        n_neighbors: int = 5
) -> pd.DataFrame:
    """
    通用缺失值填补调度函数

    Unified entry point for missing-value imputation.

    本函数作为缺失值填补的统一入口，根据 ``method`` 参数
    选择并调用对应的填补算法，对基因型矩阵中的缺失值
    执行替换操作。

    This function serves as a unified entry point for missing-value
    imputation. It dispatches to the appropriate filling algorithm
    based on the ``method`` argument and replaces missing values
    in the genotype matrix.

    Supported methods
    -----------------
    mode
        列众数填补（``_fill_mode``）。
        Column-wise mode imputation.

    mean
        列均值填补（``_fill_mean``）。
        Column-wise mean imputation.

    knn
        基于 sklearn ``KNNImputer`` 的 KNN 填补（欧氏距离）。
        KNN imputation using sklearn ``KNNImputer`` (Euclidean distance).

    knn_hamming_abs
        等权汉明距离 KNN 填补。
        Equal-weight KNN imputation using unnormalized Hamming distance.

    knn_hamming_adaptive
        自适应指数加权的汉明距离 KNN 填补。
        Adaptive weighted KNN imputation using Hamming distance.

    knn_hamming_balltree
        基于 BallTree 的汉明距离 KNN 填补。
        Hamming-distance KNN imputation accelerated with BallTree.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型矩阵，编码规则为：
        ``0`` = 参考等位基因，
        ``1`` = 变异等位基因，
        ``3`` = 缺失值。

        Genotype matrix encoded as:
        ``0`` = reference allele,
        ``1`` = alternate allele,
        ``3`` = missing value.

    method : str, default="mode"
        缺失值填补方法名称（大小写不敏感）。

        Name of the imputation method (case-insensitive).

    n_neighbors : int, default=5
        KNN 类方法使用的近邻数量，
        仅对 ``knn*`` 方法生效。

        Number of nearest neighbors used for KNN-based methods only.

    Returns
    -------
    pandas.DataFrame
        填补后的矩阵，结构与输入一致。

        Imputed DataFrame with the same structure as the input.

    Raises
    ------
    ValueError
        当 ``method`` 不是支持的方法名称时抛出。

        Raised when ``method`` is not a supported method name.

    Notes
    -----
    - 本函数首先将输入矩阵复制并转换为 ``float32``，
      然后将编码值 ``3`` 分块替换为 ``NaN``，
      以兼容后续基于 NumPy / pandas 的运算。

      The input matrix is copied and cast to ``float32``, and
      values encoded as ``3`` are replaced with ``NaN`` in
      column-wise batches for compatibility with NumPy/pandas
      operations.

    - ``3 → NaN`` 的替换过程按列分块执行（默认每块 2000 列），
      仅用于降低瞬时内存占用。

      The ``3 → NaN`` replacement is performed in column-wise
      batches (default: 2000 columns) to reduce peak memory usage.

    - 填补前后会统计并打印缺失值数量，
      但不会对残留 ``NaN`` 进行强制处理。

      The function reports missing-value counts before and after
      imputation but does not enforce removal of residual ``NaN``.

    - 具体填补逻辑由各个 ``_fill_*`` 函数实现，
      本函数仅负责方法分发与日志输出。

      The actual imputation logic is implemented in the
      corresponding ``_fill_*`` functions; this function only
      handles dispatching and logging.
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
    else:
        raise ValueError(f"Unknown filling method: {method}")

    after_nans = filled.isna().sum().sum()
    replaced = before_nans - after_nans

    if after_nans > 0:
        print(f"[WARN] There are still {after_nans} NaNs after padding (possibly an abnormal column).")
    print(f"[OK] {method.upper():<5} Filling complete | Number of missing values replaced: {replaced} | Residual NaN: {after_nans}")

    return filled


def _collapse_y_haplogroup(y: pd.Series) -> pd.Series:
    """
    将 Y 单倍群标签折叠为“字母 + 首位数字”层级

    Collapse Y haplogroup labels to the form: letter + first digit.

    本函数对 Y haplogroup 标签执行字符串级解析，
    提取其首字母，并在第二位为数字时保留该数字，
    生成一个更粗粒度的分类标签。

    This function performs string-based parsing of Y haplogroup
    labels, extracting the first letter and, if present, the
    first digit to form a coarse-grained category.

    Examples
    --------
    "R1b1a2"   → "R1"
    "R-M269"   → "R"
    "I2a1"     → "I2"
    "J2a1"     → "J2"
    "E1b1a"    → "E1"
    "C2a1"     → "C2"

    Parameters
    ----------
    y : pandas.Series
        Y haplogroup 标签序列。
        值将被转换为字符串并去除首尾空白。

        Series of Y haplogroup labels. Values are cast to strings
        and stripped of leading/trailing whitespace.

    Returns
    -------
    pandas.Series
        折叠后的 haplogroup 标签序列，
        索引与输入一致，名称为 ``<original_name>_macro2``。

        Collapsed haplogroup labels with the same index as input
        and name ``<original_name>_macro2``.

    Notes
    -----
    - 本函数不会过滤无效或未知标签
      （如 ``"n/a"``, ``"unknown"``, ``""``），
      这些值将被原样返回，并应由
      ``clean_labels_and_align`` 统一处理。

      This function does not remove invalid or unknown labels
      (e.g. ``"n/a"``, ``"unknown"``, empty strings).
      Such values are returned unchanged and should be handled
      by ``clean_labels_and_align``.

    - 仅当标签以字母开头时才进行解析；
      否则该值将被原样返回。

      Parsing is applied only when the label starts with a letter;
      otherwise, the value is returned unchanged.

    - 若第二个字符不是数字，
      则结果退化为单字母宏类（如 ``"R-M269" → "R"``）。

      If the second character is not a digit, the label is
      collapsed to a single-letter category.
    """

    y = y.astype(str).str.strip()

    macro = []

    for v in y:
        v = v.strip()

        # 空值：原样返回，由 clean_labels_and_align 统一过滤
        if not v:
            macro.append(v)
            continue

        # 必须以字母开头，否则原样返回
        if not v[0].isalpha():
            macro.append(v)
            continue

        first = v[0].upper()

        # 第二位是数字 → 字母 + 数字（R1, I2, N1）
        if len(v) > 1 and v[1].isdigit():
            macro.append(first + v[1])
        else:
            # 没有数字 → 回退到单字母（R, J, E）
            macro.append(first)

    return pd.Series(macro, index=y.index, name=y.name + "_macro2")


def clean_labels_and_align(
        emb: pd.DataFrame,
        labels: pd.Series,
        collapse_y: bool = False,
        invalid_values=None,
        whitelist=None,
):
    """
    清洗分类标签并与 embedding 行对齐

    Clean categorical labels and align them with the embedding matrix.

    本函数对分类标签进行标准化与过滤，
    并根据过滤后的有效标签索引，同步筛选并对齐
    降维后的 embedding 矩阵，确保二者在样本维度上的一致性。

    This function cleans and filters categorical labels, then
    aligns the embedding matrix to the retained labels so that
    both outputs correspond to the same set of samples.

    Parameters
    ----------
    emb : pandas.DataFrame
        降维后的 embedding 矩阵（2D 或 ND），
        行表示样本，行索引为样本 ID。

        Embedding matrix (2D or ND) with samples as rows and
        sample IDs as the index.

    labels : pandas.Series
        分类标签序列，索引与 ``emb`` 的行索引对应。

        Categorical label series indexed by sample IDs.

    collapse_y : bool, default=False
        是否将 Y 染色体单倍群标签映射为更高层级的大类。
        仅在标签为 Y haplogroup 时生效。

        Whether to collapse Y haplogroup labels into
        higher-level categories.

    invalid_values : iterable or None, default=None
        需要被视为无效并过滤掉的标签集合。
        若为 ``None``，则使用函数内部定义的默认无效值集合。

        Label values to be treated as invalid and removed.
        If ``None``, a predefined default set is used.

    whitelist : iterable or None, default=None
        若提供，仅保留该白名单中的标签类别（不区分大小写）。

        If provided, only labels contained in this whitelist
        (case-insensitive) are retained.

    Returns
    -------
    emb_clean : pandas.DataFrame
        过滤并对齐后的 embedding 矩阵，
        行索引已重置为连续整数索引。

        Filtered and aligned embedding matrix with
         a reset integer index.

    labels_clean : pandas.Series
        与 ``emb_clean`` 行顺序一致的标签序列，
        行索引已重置。

        Cleaned label series aligned with ``emb_clean``.

    Notes
    -----
    - 标签在比较前会被统一转换为字符串并去除首尾空白，
      同时使用小写形式进行无效值与白名单匹配。

      Labels are cast to strings, stripped of whitespace,
      and compared in lowercase form for invalid-value and
      whitelist filtering.

    - 无效标签的过滤基于字符串匹配，
      不依赖于标签的原始数据类型。

      Invalid labels are removed via string-based matching,
      independent of the original data type.

    - 过滤完成后，embedding 与 labels 都会
      使用相同的有效样本索引进行筛选，
      并调用 ``reset_index(drop=True)`` 重建索引。

      After filtering, both embedding and labels are subset
      using the same valid sample indices and have their
      indices reset via ``reset_index(drop=True)``.

    - 若 ``collapse_y=True``，
      会在标签过滤完成后调用内部函数
      ``_collapse_y_haplogroup`` 对标签进一步映射。

      When ``collapse_y=True``, the internal function
      ``_collapse_y_haplogroup`` is applied after filtering.
    """

    # 默认无效类别（可扩展）
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

    # 全部转成 str，并 strip 空格
    labels_clean = labels.astype(str).str.strip()

    # 全部转成小写，便于比较
    normalized = labels_clean.str.lower()

    # 将 invalid 映射为统一小写
    invalid_lower = {v.lower() for v in invalid_values}

    # 标记有效标签
    mask_valid = ~normalized.isin(invalid_lower)

    # 若有 whitelist，进一步过滤
    if whitelist is not None:
        whitelist_lower = {w.lower() for w in whitelist}
        mask_valid &= normalized.isin(whitelist_lower)

    # 根据 valid index 过滤 embedding
    valid_idx = labels.index[mask_valid]

    emb_clean = emb.loc[valid_idx].reset_index(drop=True)
    labels_clean = labels_clean.loc[valid_idx].reset_index(drop=True)

    if collapse_y:
        labels_clean = _collapse_y_haplogroup(labels_clean)

    return emb_clean, labels_clean