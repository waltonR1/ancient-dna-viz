"""
embedding.py
-----------------
降维与嵌入模块（Embedding Layer）
Dimensionality reduction and embedding module.

用于将高维基因型矩阵投影到低维空间（2D 或 3D），
以便进行可视化与聚类分析。
支持多种主流降维算法（UMAP、t-SNE、MDS、Isomap），
并提供基于增量 PCA 与 Parquet 分片的伪流式 UMAP 降维实现。
Used for projecting high-dimensional genotype matrices into lower-dimensional spaces (2D or 3D)
for visualization and clustering analysis.
Supports multiple mainstream embedding algorithms (UMAP, t-SNE, MDS, Isomap)
and includes a streaming-like UMAP implementation using incremental PCA and Parquet shards.

功能 / Functions:
    - _compute_umap(): 执行 UMAP 降维。
      Perform UMAP dimensionality reduction.
    - _compute_tsne(): 执行 t-SNE 降维。
      Perform t-SNE dimensionality reduction.
    - _compute_mds(): 执行 MDS 降维。
      Perform MDS dimensionality reduction.
    - _compute_isomap(): 执行 Isomap 降维。
      Perform Isomap manifold embedding.
    - compute_embeddings(): 降维统一接口，根据 method 自动调度。
      Unified interface that dispatches to the chosen reduction algorithm.
    - streaming_umap_from_parquet(): 流式 UMAP 降维，适用于大规模数据集。
      Streaming-like UMAP with incremental PCA for large datasets.

说明 / Description:
    本模块是整个管线的“嵌入层”（Embedding Layer），
    负责将高维基因型矩阵压缩为可解释的低维空间表示。
    输出结果与样本顺序完全一致，适用于后续聚类或可视化。
    This module serves as the embedding layer of the pipeline,
    transforming high-dimensional genotype matrices into interpretable lower-dimensional spaces.
    Output coordinates maintain exact sample ordering for downstream clustering and visualization.
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

def _compute_umap(X: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    执行 UMAP 降维
    Perform UMAP dimensionality reduction.

    :param X: pd.DataFrame
        基因型矩阵（行 = 样本，列 = SNP）。
        Genotype matrix (rows = samples, columns = SNPs).

    :param n_components: int, default=2
        降维目标维度。
        Target number of dimensions.

    :param kwargs: dict
        传递给 UMAP 的额外参数（如 random_state, metric 等）。
        Additional arguments passed to the UMAP constructor.

    :return: pd.DataFrame
        降维结果 DataFrame，列名为 ["Dim1", "Dim2", ...]。
        Reduced DataFrame with columns ["Dim1", "Dim2", ...].

    说明 / Notes:
        - 支持 random_state 参数以保证可重复性。
          Supports `random_state` for reproducibility.
        - 能较好地保留全局与局部簇结构。
          Preserves both global and local cluster structures.
        - 输出顺序与输入样本一致。
          Output sample order is identical to input.
    """
    model = UMAP(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_tsne(X: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    执行 t-SNE 降维
    Perform t-SNE dimensionality reduction.

    :param X: pd.DataFrame
        基因型矩阵。
        Genotype matrix.

    :param n_components: int, default=2
        降维目标维度。
        Target number of dimensions.

    :param kwargs: dict
        传递给 TSNE 的额外参数。
        Additional parameters passed to TSNE.

    :return: pd.DataFrame
        降维结果 DataFrame。
        DataFrame containing reduced coordinates.

    说明 / Notes:
        - 支持 random_state。
          Supports `random_state`.
        - 使用欧氏距离，适合局部结构可视化。
          Uses Euclidean distance; suitable for local structure visualization.
        - 计算复杂度较高，不推荐超大数据集使用。
          Computationally expensive; not ideal for very large datasets.
    """
    model = TSNE(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_mds(X: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    执行 MDS 降维
    Perform MDS dimensionality reduction.

    :param X: pd.DataFrame
        基因型矩阵。
        Genotype matrix.

    :param n_components: int, default=2
        降维目标维度。
        Target number of dimensions.

    :param kwargs: dict
        传递给 MDS 的额外参数。
        Additional parameters passed to MDS.

    :return: pd.DataFrame
        降维结果 DataFrame。
        Reduced DataFrame.

    说明 / Notes:
        - 不支持 random_state。
          Does not support `random_state`.
        - 适合线性结构可视化。
          Suitable for linear-structure visualization.
        - 计算复杂度较高。
          Computationally intensive.
    """
    kwargs.pop("random_state", None)
    model = MDS(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_isomap(X: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    执行 Isomap 降维
    Perform Isomap manifold embedding.

    :param X: pd.DataFrame
        基因型矩阵。
        Genotype matrix.

    :param n_components: int, default=2
        降维目标维度。
        Target number of dimensions.

    :param kwargs: dict
        传递给 Isomap 的额外参数。
        Additional parameters passed to Isomap.

    :return: pd.DataFrame
        降维结果 DataFrame。
        Reduced DataFrame.

    说明 / Notes:
        - 不支持 random_state。
          Does not support `random_state`.
        - 适合非线性流形结构的学习与可视化。
          Preserves global manifold structure for nonlinear embeddings.
    """
    kwargs.pop("random_state", None)
    model = Isomap(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def compute_embeddings(X: pd.DataFrame, method: str = "umap", n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    降维统一接口
    Unified interface for dimensionality reduction.

    根据指定 method 自动选择对应算法执行降维。
    Automatically dispatches to the chosen embedding algorithm.

    :param X: pd.DataFrame
        基因型矩阵。
        Genotype matrix.

    :param method: str, default="umap"
        降维方法（"umap"、"tsne"、"mds"、"isomap"）。
        Embedding method ("umap", "tsne", "mds", "isomap").

    :param n_components: int, default=2
        目标维度（2 或 3）。
        Target number of dimensions (2 or 3).

    :param kwargs: dict
        传递给具体算法的额外参数。
        Extra parameters passed to the underlying algorithm.

    :return: pd.DataFrame
        投影后的 DataFrame。
        Projected DataFrame.

    说明 / Notes:
        - 根据 method 自动调用相应算法。
          Automatically dispatches to the proper method.
        - 输出列名统一为 ["Dim1", "Dim2", ...]。
          Output columns standardized as ["Dim1", "Dim2", ...].
        - 对不支持 random_state 的算法自动忽略该参数。
          Random state parameter is ignored for unsupported methods.
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
        raise ValueError(f"未知降维方法: {method}")

    print(f"[OK] Embeddings computed with method: {method}")
    return embedding


def streaming_umap_from_parquet(dataset_dir: str | Path, n_components: int = 2, max_cols: int = 50000, pca_dim: int = 50, random_state: int = 42,) -> pd.DataFrame:
    """
    Streaming-like UMAP（伪流式降维）
    Streaming-like UMAP dimensionality reduction.

    通过增量 PCA 与分片 Parquet 文件实现低内存占用的降维流程。
    Combines IncrementalPCA and Parquet-shard streaming to perform
    low-memory UMAP embedding for very large datasets.

    :param dataset_dir: str | Path
        分片目录路径（包含 columns_index.json）。
        Directory containing Parquet shards and `columns_index.json`.

    :param n_components: int, default=2
        UMAP 降维维数。
        Target UMAP dimensionality.

    :param max_cols: int, default=50000
        每个分片最多读取的列数。
        Maximum number of columns to read per shard.

    :param pca_dim: int, default=50
        PCA 压缩维度。
        Number of PCA components retained before UMAP.

    :param random_state: int, default=42
        随机种子。
        Random seed.

    :return: pd.DataFrame
        最终降维结果。
        Final embedding DataFrame.

    说明 / Notes:
        - 从 columns_index.json 加载列分片索引。
          Loads shard metadata from `columns_index.json`.
        - 使用 IncrementalPCA 实现逐分片拟合与转换，避免内存峰值。
          Fits and transforms shards incrementally using IncrementalPCA to avoid memory spikes.
        - 拼接所有分片的 PCA 输出后执行 UMAP 降维。
          Concatenates reduced outputs from all shards before final UMAP embedding.
        - 适合超大规模基因型矩阵的可视化与特征压缩任务。
          Designed for extremely large genotype datasets where full in-memory UMAP is infeasible.
    """
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "columns_index.json"

    # === Step 0: 检查元数据 ===
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Missing columns_index.json in {dataset_dir}. "
            f"Please regenerate it during mode filling."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        col_index_meta = json.load(f)

    print(f"[STREAM-UMAP] Loaded column metadata for {len(col_index_meta)} shards")

    # === Step 1: Incremental PCA 训练阶段 ===
    ipca = IncrementalPCA(n_components=pca_dim)
    expected_features = None  # 第一次确定特征维度

    for meta in tqdm(col_index_meta, desc="[STREAM] Incremental PCA fitting"):
        part_path = dataset_dir / meta["part"]
        cols = meta["columns"][:max_cols]
        table = pq.read_table(part_path, columns=cols)
        X = table.to_pandas().fillna(1.0)
        X_np = X.to_numpy(dtype=np.float32)

        # 第一次设定标准列数
        if expected_features is None:
            expected_features = X_np.shape[1]
            print(f"[INFO] Reference feature size: {expected_features}")
        elif X_np.shape[1] < expected_features:
            # 若最后一个分片列较少，用 1.0 填补差值
            pad_width = expected_features - X_np.shape[1]
            X_np = np.pad(X_np, ((0, 0), (0, pad_width)), constant_values=1.0)

        ipca.partial_fit(X_np)

    # === Step 2: PCA 转换阶段 ===
    partial_embeddings = []
    for meta in tqdm(col_index_meta, desc="[STREAM] Incremental PCA transform"):
        part_path = dataset_dir / meta["part"]
        cols = meta["columns"][:max_cols]
        table = pq.read_table(part_path, columns=cols)
        X = table.to_pandas().fillna(1.0)
        X_np = X.to_numpy(dtype=np.float32)

        # 若特征维度不足，补齐
        if X_np.shape[1] < expected_features:
            X_np = np.pad(X_np, ((0, 0), (0, expected_features - X_np.shape[1])), constant_values=1.0)

        X_red = ipca.transform(X_np)
        partial_embeddings.append(X_red)

    # === Step 3: 拼接所有 PCA 结果 ===
    X_all = np.hstack(partial_embeddings)
    print(f"[OK] Incremental PCA complete → {X_all.shape}")

    # === Step 4: UMAP 降维 ===
    print(f"[UMAP] Running final UMAP on compressed data...")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=n_components,
        random_state=random_state,
        metric="euclidean"
    )
    emb = umap_model.fit_transform(X_all)
    emb = pd.DataFrame(emb, columns=[f"Dim{i+1}" for i in range(n_components)])

    print(f"[OK] Streaming-like UMAP complete → {emb.shape}")
    return emb
