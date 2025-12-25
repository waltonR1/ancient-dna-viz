"""
embedding.py
============

降维与嵌入模块（Embedding Layer）
Dimensionality reduction and embedding module.

本模块用于将高维基因型矩阵投影到低维空间（2D 或 3D），
以便进行可视化与后续聚类分析。
支持多种主流降维与流形学习算法，包括 UMAP、t-SNE、
MDS 和 Isomap。

This module projects high-dimensional genotype matrices into
low-dimensional spaces (2D or 3D) for visualization and
downstream clustering analysis. It supports multiple
dimensionality reduction and manifold learning algorithms,
including UMAP, t-SNE, MDS, and Isomap.

Functions
---------
_compute_umap
    执行 UMAP 降维。
    Perform UMAP dimensionality reduction.

_compute_tsne
    执行 t-SNE 降维。
    Perform t-SNE dimensionality reduction.

_compute_mds
    执行 MDS 降维。
    Perform MDS dimensionality reduction.

_compute_isomap
    执行 Isomap 流形嵌入。
    Perform Isomap manifold embedding.

compute_embeddings
    降维统一接口，根据指定方法自动调度。
    Unified interface that dispatches to the selected
    dimensionality reduction algorithm.

streaming_umap_from_parquet
    基于增量 PCA 与 Parquet 分片的伪流式 UMAP 降维，
    适用于超大规模数据集。
    Streaming-like UMAP embedding using incremental PCA
    and Parquet shards for very large datasets.

Description
-----------
本模块是整个分析流程中的“嵌入层（Embedding Layer）”，
负责将高维基因型矩阵压缩为可解释的低维空间表示。
所有输出坐标严格保持样本顺序一致，
可直接用于后续聚类或可视化分析。

This module serves as the embedding layer of the analysis
pipeline, transforming high-dimensional genotype matrices
into interpretable low-dimensional representations.
Output coordinates strictly preserve sample ordering for
downstream clustering and visualization.
"""

from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE, MDS, Isomap
from tqdm import tqdm
from umap import UMAP
from sklearn.decomposition import IncrementalPCA
import numpy as np
import json
import pyarrow.parquet as pq


def _compute_umap(
        X: pd.DataFrame,
        n_components: int = 2,
        **kwargs
) -> pd.DataFrame:
    """
    执行 UMAP 降维（内部函数）

    Perform UMAP dimensionality reduction (internal helper).

    本函数使用 UMAP 算法将高维基因型特征矩阵
    投影到低维嵌入空间（通常为 2D 或 3D），
    主要用于后续的可视化与聚类分析。

    This function applies the UMAP algorithm to project
    high-dimensional genotype features into a low-dimensional
    embedding space (typically 2D or 3D), mainly for
    visualization and downstream clustering.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型特征矩阵，行表示样本，列表示 SNP 特征。

        Genotype feature matrix where rows correspond to samples
        and columns correspond to SNP features.

    n_components : int, default=2
        降维后的目标维度。

        Target number of embedding dimensions.

    **kwargs : dict
        传递给 ``umap.UMAP`` 构造函数的额外参数
        （如 ``random_state``、``metric``、``n_neighbors`` 等）。

        Additional keyword arguments passed to the
        ``umap.UMAP`` constructor.

    Returns
    -------
    pandas.DataFrame
        降维后的嵌入坐标矩阵，列名统一为
        ``["Dim1", "Dim2", ...]``，顺序与输入样本一致。

        DataFrame containing embedding coordinates with
        standardized column names ``["Dim1", "Dim2", ...]``,
        preserving the input sample order.

    Notes
    -----
    - 本函数在内部对缺失值进行简单填补（使用 0），
      以满足 UMAP 对数值输入的要求。

      Missing values are internally filled with zeros
      before applying UMAP.

    - 支持通过 ``random_state`` 参数控制结果的可重复性。

      Reproducibility can be ensured by passing
      ``random_state`` via ``kwargs``.

    - 该函数为内部工具函数，通常不直接对外调用。

      This function is an internal utility and is not
      intended to be called directly by users.
    """

    model = UMAP(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_tsne(
        X: pd.DataFrame,
        n_components: int = 2,
        **kwargs
) -> pd.DataFrame:
    """
    执行 t-SNE 降维（内部函数）

    Perform t-SNE dimensionality reduction (internal helper).

    本函数使用 t-SNE（t-distributed Stochastic Neighbor Embedding）
    算法将高维基因型特征矩阵投影到低维嵌入空间，
    主要用于探索样本的局部结构与相似性。

    This function applies t-SNE (t-distributed Stochastic Neighbor
    Embedding) to project high-dimensional genotype features into
    a low-dimensional embedding space, mainly for exploring
    local neighborhood structure and sample similarity.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型特征矩阵，行表示样本，列表示 SNP 特征。

        Genotype feature matrix where rows correspond to samples
        and columns correspond to SNP features.

    n_components : int, default=2
        降维后的目标维度。

        Target number of embedding dimensions.

    **kwargs : dict
        传递给 ``sklearn.manifold.TSNE`` 构造函数的额外参数
        （如 ``perplexity``、``learning_rate``、``random_state`` 等）。

        Additional keyword arguments passed to the
        ``sklearn.manifold.TSNE`` constructor.

    Returns
    -------
    pandas.DataFrame
        降维后的嵌入坐标矩阵，列名统一为
        ``["Dim1", "Dim2", ...]``，顺序与输入样本一致。

        DataFrame containing embedding coordinates with
        standardized column names ``["Dim1", "Dim2", ...]``,
        preserving the input sample order.

    Notes
    -----
    - 本函数在内部对缺失值进行简单填补（使用 0），
      以满足 t-SNE 对数值输入的要求。

      Missing values are internally filled with zeros
      before applying t-SNE.

    - t-SNE 主要保留局部结构，不保证全局距离关系。

      t-SNE emphasizes local neighborhood structure and
      does not preserve global distances.

    - 该算法计算复杂度较高，不适合超大规模数据集。

      t-SNE is computationally expensive and not recommended
      for very large datasets.

    - 该函数为内部工具函数，通常不直接对外调用。

      This function is an internal utility and is not
      intended to be called directly by users.
    """

    model = TSNE(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_mds(
        X: pd.DataFrame,
        n_components: int = 2,
        **kwargs
) -> pd.DataFrame:
    """
    执行 MDS 降维（内部函数）

    Perform MDS dimensionality reduction (internal helper).

    本函数使用 MDS（Multidimensional Scaling，多维尺度分析）
    算法将高维基因型特征矩阵映射到低维空间，
    主要用于保持样本之间的整体距离结构。

    This function applies Multidimensional Scaling (MDS) to
    project high-dimensional genotype features into a
    low-dimensional space while preserving pairwise distances
    as faithfully as possible.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型特征矩阵，行表示样本，列表示 SNP 特征。

        Genotype feature matrix where rows correspond to samples
        and columns correspond to SNP features.

    n_components : int, default=2
        降维后的目标维度。

        Target number of embedding dimensions.

    **kwargs : dict
        传递给 ``sklearn.manifold.MDS`` 构造函数的额外参数。
        注意：``random_state`` 参数将被自动忽略。

        Additional keyword arguments passed to the
        ``sklearn.manifold.MDS`` constructor.
        Note that the ``random_state`` parameter is ignored.

    Returns
    -------
    pandas.DataFrame
        降维后的嵌入坐标矩阵，列名统一为
        ``["Dim1", "Dim2", ...]``，顺序与输入样本一致。

        DataFrame containing embedding coordinates with
        standardized column names ``["Dim1", "Dim2", ...]``,
        preserving the input sample order.

    Notes
    -----
    - 本函数在内部对缺失值进行简单填补（使用 0），
      以满足 MDS 对数值输入的要求。

      Missing values are internally filled with zeros
      before applying MDS.

    - 本实现中会自动移除 ``random_state`` 参数，
      因为 ``sklearn.manifold.MDS`` 不支持该参数。

      The ``random_state`` parameter is explicitly removed
      because it is not supported by ``sklearn.manifold.MDS``.

    - MDS 计算复杂度较高，适用于样本规模较小的情况。

      MDS is computationally expensive and best suited
      for relatively small datasets.

    - 该函数为内部工具函数，通常不直接对外调用。

      This function is an internal utility and is not
      intended to be called directly by users.
    """

    kwargs.pop("random_state", None)
    model = MDS(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_isomap(
        X: pd.DataFrame,
        n_components: int = 2,
        **kwargs
) -> pd.DataFrame:
    """
    执行 Isomap 流形嵌入（内部函数）

    Perform Isomap manifold embedding (internal helper).

    本函数使用 Isomap（Isometric Mapping）算法
    将高维基因型特征矩阵映射到低维嵌入空间，
    通过近邻图上的测地距离来刻画样本之间的全局流形结构。

    This function applies the Isomap algorithm to project
    high-dimensional genotype features into a low-dimensional
    embedding space, preserving global manifold structure
    via geodesic distances on a neighborhood graph.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型特征矩阵，行表示样本，列表示 SNP 特征。

        Genotype feature matrix where rows correspond to samples
        and columns correspond to SNP features.

    n_components : int, default=2
        降维后的目标维度。

        Target number of embedding dimensions.

    **kwargs : dict
        传递给 ``sklearn.manifold.Isomap`` 构造函数的额外参数。
        注意：``random_state`` 参数将被自动忽略。

        Additional keyword arguments passed to the
        ``sklearn.manifold.Isomap`` constructor.
        Note that the ``random_state`` parameter is ignored.

    Returns
    -------
    pandas.DataFrame
        降维后的嵌入坐标矩阵，列名统一为
        ``["Dim1", "Dim2", ...]``，顺序与输入样本一致。

        DataFrame containing embedding coordinates with
        standardized column names ``["Dim1", "Dim2", ...]``,
        preserving the input sample order.

    Notes
    -----
    - 本函数在内部对缺失值进行简单填补（使用 0），
      以满足 Isomap 对数值输入的要求。

      Missing values are internally filled with zeros
      before applying Isomap.

    - 本实现中会自动移除 ``random_state`` 参数，
      因为 ``sklearn.manifold.Isomap`` 不支持该参数。

      The ``random_state`` parameter is explicitly removed
      because it is not supported by ``sklearn.manifold.Isomap``.

    - Isomap 更侧重保持样本之间的全局流形结构，
      相比 t-SNE 更强调全局距离一致性。

      Isomap emphasizes preservation of global manifold
      structure rather than purely local neighborhoods.

    - 该函数为内部工具函数，通常不直接对外调用。

      This function is an internal utility and is not
      intended to be called directly by users.
    """

    kwargs.pop("random_state", None)
    model = Isomap(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def compute_embeddings(
        X: pd.DataFrame,
        method: str = "umap",
        n_components: int = 2,
        **kwargs
) -> pd.DataFrame:
    """
    降维与嵌入的统一接口

    Unified interface for dimensionality reduction and embedding.

    本函数根据指定的 ``method`` 参数，
    自动调度对应的降维或流形学习算法，
    将高维基因型特征矩阵映射到低维嵌入空间。

    This function dispatches to the selected dimensionality
    reduction or manifold learning algorithm based on the
    specified ``method`` and projects high-dimensional genotype
    features into a low-dimensional embedding space.

    Parameters
    ----------
    X : pandas.DataFrame
        基因型特征矩阵，行表示样本，列表示 SNP 特征。

        Genotype feature matrix where rows correspond to samples
        and columns correspond to SNP features.

    method : str, default="umap"
        降维方法名称，可选值包括：
        ``"umap"``, ``"tsne"``, ``"mds"``, ``"isomap"``。

        Embedding method to use. Supported values include
        ``"umap"``, ``"tsne"``, ``"mds"``, and ``"isomap"``.

    n_components : int, default=2
        降维后的目标维度（通常为 2 或 3）。

        Target number of embedding dimensions (typically 2 or 3).

    **kwargs : Any
        传递给具体降维算法构造函数的额外参数。
        具体支持的参数取决于所选算法。

        Additional keyword arguments passed to the underlying
        embedding algorithm. Supported parameters depend on
        the selected method.

    Returns
    -------
    pandas.DataFrame
        降维后的嵌入坐标矩阵，列名统一为
        ``["Dim1", "Dim2", ...]``，顺序与输入样本一致。

        DataFrame containing embedding coordinates with
        standardized column names ``["Dim1", "Dim2", ...]``,
        preserving the input sample order.

    Notes
    -----
    - 本函数会根据 ``method`` 自动调用对应的内部实现函数。

      The function automatically dispatches to the corresponding
      internal implementation based on ``method``.

    - 对于不支持 ``random_state`` 参数的算法（如 MDS、Isomap），
      该参数会被自动忽略。

      For algorithms that do not support ``random_state``
      (e.g. MDS and Isomap), the parameter is silently ignored.

    - 输出坐标严格保持与输入样本顺序一致，
      可直接用于后续聚类或可视化分析。

      Output coordinates strictly preserve the input sample
      order and can be used directly for downstream clustering
      or visualization.

    Raises
    ------
    ValueError
        当指定的 ``method`` 不被支持时抛出。

        Raised when the specified embedding method is not supported.
    """

    print(f"[INFO] Compute embeddings with method: {method}")
    method = method.lower()

    if method == "umap":
        embedding = _compute_umap(X, n_components=n_components, **kwargs)
    elif method == "tsne":
        embedding = _compute_tsne(X, n_components=n_components, **kwargs)
    elif method == "mds":
        embedding = _compute_mds(X, n_components=n_components, **kwargs)
    elif method == "isomap":
        embedding = _compute_isomap(X, n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknow method: {method}")

    print(f"[OK] Embeddings computed with method: {method}")
    return embedding


def streaming_umap_from_parquet(
        dataset_dir: str | Path,
        n_components: int = 2,
        max_cols: int = 50000,
        pca_dim: int = 50,
        random_state: int = 42,
) -> pd.DataFrame:
    """
    基于 Parquet 分片的伪流式 UMAP 降维

    Streaming-like UMAP dimensionality reduction from Parquet shards.

    本函数面向超大规模基因型矩阵，
    通过「分片读取 + 增量 PCA + 最终 UMAP」的方式，
    在有限内存条件下实现近似流式的 UMAP 降维。

    This function targets extremely large genotype matrices
    and performs streaming-like UMAP embedding by combining
    Parquet shard loading, Incremental PCA, and a final UMAP
    projection under limited memory conditions.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        包含 Parquet 分片文件和 ``columns_index.json`` 的目录路径。

        Directory containing Parquet shards and
        the ``columns_index.json`` metadata file.

    n_components : int, default=2
        最终 UMAP 嵌入空间的维度。

        Target dimensionality of the final UMAP embedding.

    max_cols : int, default=50000
        每个分片中最多读取的特征列数。

        Maximum number of feature columns loaded per shard.

    pca_dim : int, default=50
        在执行 UMAP 之前，通过 Incremental PCA
        压缩到的中间特征维度。

        Number of PCA components retained before applying UMAP.

    random_state : int, default=42
        随机种子，用于保证 UMAP 结果的可重复性。

        Random seed for reproducibility of the final UMAP embedding.

    Returns
    -------
    pandas.DataFrame
        最终降维后的嵌入坐标矩阵，列名统一为
        ``["Dim1", "Dim2", ...]``，顺序与原始样本一致。

        DataFrame containing the final embedding coordinates
        with standardized column names, preserving sample order.

    Notes
    -----
    - 本函数依赖 ``columns_index.json`` 文件来描述
      每个 Parquet 分片所包含的特征列。

      The function relies on ``columns_index.json`` to describe
      the feature columns contained in each Parquet shard.

    - 降维流程分为四个阶段：

      The dimensionality reduction pipeline consists of four stages:

      1) 增量 PCA 拟合（Incremental PCA fitting）
      2) 增量 PCA 转换（Incremental PCA transform）
      3) 拼接所有分片的 PCA 输出
      4) 在压缩特征空间上执行最终 UMAP

    - 该方法并非真正的在线（online）UMAP，
      而是一种内存友好的伪流式实现。

      This approach is not a true online UMAP algorithm,
      but a memory-efficient, streaming-like approximation.

    - 适用于完整 UMAP 无法在内存中执行的超大规模数据集。

      Designed for extremely large datasets where
      full in-memory UMAP is infeasible.

    Raises
    ------
    FileNotFoundError
        当 ``columns_index.json`` 文件不存在时抛出。

        Raised when ``columns_index.json`` is missing.
    """

    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "columns_index.json"

    # 检查元数据
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Missing columns_index.json in {dataset_dir}. "
            f"Please regenerate it during mode filling."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        col_index_meta = json.load(f)

    print(f"[STREAM-UMAP] Loaded column metadata for {len(col_index_meta)} shards")

    # Incremental PCA 训练阶段
    ipca = IncrementalPCA(n_components=pca_dim)
    expected_features = None  # 第一次确定特征维度

    for meta in tqdm(col_index_meta, desc="[STREAM] Incremental PCA fitting"):
        part_path = dataset_dir / meta["part"]
        cols = meta["columns"][:max_cols]
        table = pq.read_table(part_path, columns=cols)
        cols_np = [c.to_numpy(zero_copy_only=False) for c in table.columns]
        X_np = np.column_stack(cols_np).astype(np.float32)
        X_np = np.nan_to_num(X_np, nan=1.0)

        # 第一次设定标准列数
        if expected_features is None:
            expected_features = X_np.shape[1]
            print(f"[INFO] Reference feature size: {expected_features}")
        elif X_np.shape[1] < expected_features:
            # 若最后一个分片列较少，用 1.0 填补差值
            pad_width = expected_features - X_np.shape[1]
            X_np = np.pad(X_np, ((0, 0), (0, pad_width)), constant_values=1.0)

        ipca.partial_fit(X_np)

    # PCA 转换阶段
    partial_embeddings = []
    for meta in tqdm(col_index_meta, desc="[STREAM] Incremental PCA transform"):
        part_path = dataset_dir / meta["part"]
        cols = meta["columns"][:max_cols]
        table = pq.read_table(part_path, columns=cols)
        cols_np = [c.to_numpy(zero_copy_only=False) for c in table.columns]
        X_np = np.column_stack(cols_np).astype(np.float32)
        X_np = np.nan_to_num(X_np, nan=1.0)

        # 若特征维度不足，补齐
        if X_np.shape[1] < expected_features:
            X_np = np.pad(X_np, ((0, 0), (0, expected_features - X_np.shape[1])), constant_values=1.0)

        X_red = ipca.transform(X_np)
        partial_embeddings.append(X_red)

    # 拼接所有 PCA 结果
    X_all = np.hstack(partial_embeddings)
    print(f"[OK] Incremental PCA complete → {X_all.shape}")

    # UMAP 降维
    print(f"[UMAP] Running final UMAP on compressed data...")
    umap_model = UMAP(
        n_neighbors=50,
        min_dist=0.3,
        metric="hamming",
        n_components=n_components,
        random_state=random_state
    )
    emb = umap_model.fit_transform(X_all)
    emb = pd.DataFrame(emb, columns=[f"Dim{i+1}" for i in range(n_components)])

    print(f"[OK] Streaming-like UMAP complete → {emb.shape}")
    return emb
