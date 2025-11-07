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
    UMAP 降维。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param n_components: 降维目标维度（默认 2）。
    :param kwargs: 传递给 UMAP 的额外参数。
    :return: 降维结果 DataFrame，列名为 ["Dim1", "Dim2", ...]。
    说明:
        - 支持 random_state；
        - 适合保持全局结构与局部簇结构；
        - 输出结构与样本顺序保持一致。
    """
    model = UMAP(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_tsne(X: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    t-SNE 降维。

    :param X: 基因型矩阵 (pd.DataFrame)。
    :param n_components: 降维目标维度（默认 2）。
    :param kwargs: 传递给 TSNE 的额外参数。
    :return: 降维结果 DataFrame。
    说明:
        - 支持 random_state；
        - 仅支持欧氏距离；
        - 适合局部结构可视化。
    """
    model = TSNE(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_mds(X: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    MDS 降维。

    :param X: 基因型矩阵 (pd.DataFrame)。
    :param n_components: 降维目标维度（默认 2）。
    :param kwargs: 传递给 MDS 的额外参数。
    :return: 降维结果 DataFrame。
    说明:
        - 不支持 random_state；
        - 适合线性结构可视化；
        - 计算复杂度较高。
    """
    kwargs.pop("random_state", None)
    model = MDS(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def _compute_isomap(X: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    Isomap 降维。

    :param X: 基因型矩阵 (pd.DataFrame)。
    :param n_components: 降维目标维度（默认 2）。
    :param kwargs: 传递给 Isomap 的额外参数。
    :return: 降维结果 DataFrame。
    说明:
        - 不支持 random_state；
        - 适合流形学习任务；
        - 保留非线性结构的全局嵌入。
    """
    kwargs.pop("random_state", None)
    model = Isomap(n_components=n_components, **kwargs)
    coords = model.fit_transform(X.fillna(0))
    return pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])


def compute_embeddings(X: pd.DataFrame, method: str = "umap", n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    降维统一接口。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param method: 降维方法（"umap" / "tsne" / "mds" / "isomap"）。
    :param n_components: 目标维度（2 或 3），默认 2。
    :param kwargs: 传递给具体算法的额外参数。
    :return: 投影后的 DataFrame。
    说明:
        - 自动根据 method 调用对应算法；
        - 输出列名为 ["Dim1", "Dim2", ...]；
        - 若算法不支持 random_state，会自动忽略。
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


def streaming_umap_from_parquet(
    dataset_dir: str | Path,
    n_components: int = 2,
    max_cols: int = 50000,
    pca_dim: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Streaming-like UMAP（伪流式降维）
    ===========================================================
    步骤：
      1. 读取列索引元数据 columns_index.json；
      2. 按每个分片的真实列加载；
      3. 增量训练 IncrementalPCA；
      4. 拼接 PCA 结果后执行 UMAP；
      5. 返回二维嵌入坐标。

    :param dataset_dir: 分片目录路径
    :param n_components: UMAP 降维维数
    :param max_cols: 每个分片最多读取的特征列
    :param pca_dim: PCA 压缩维度
    :param random_state: 随机种子
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

    # === Step 0.5: 构建全局列空间（master_cols） ===
    master_cols = []
    for meta in col_index_meta:
        for c in meta["columns"]:
            if c not in master_cols:
                master_cols.append(c)
            if len(master_cols) >= max_cols:
                break
        if len(master_cols) >= max_cols:
            break
    print(f"[INFO] Unified master column space: {len(master_cols)} features")

    # === Step 1: Incremental PCA 训练阶段 ===
    ipca = IncrementalPCA(n_components=pca_dim)
    col_to_idx = {c: i for i, c in enumerate(master_cols)}
    for meta in tqdm(col_index_meta, desc="[STREAM] Incremental PCA fitting"):
        part_path = dataset_dir / meta["part"]
        cols = [c for c in meta["columns"] if c in master_cols]
        table = pq.read_table(part_path, columns=cols)
        X = table.to_pandas().fillna(1.0)

        arr = np.full((len(X), len(master_cols)), 1.0, dtype=np.float32)
        for j, c in enumerate(cols):
            arr[:, col_to_idx[c]] = X[c].to_numpy(dtype=np.float32)
        ipca.partial_fit(arr)

    # === Step 2: PCA 转换阶段 ===
    partial_embeddings = []
    for meta in tqdm(col_index_meta, desc="[STREAM] Incremental PCA transform"):
        part_path = dataset_dir / meta["part"]
        cols = [c for c in meta["columns"] if c in master_cols]
        table = pq.read_table(part_path, columns=cols)
        X = table.to_pandas().fillna(1.0)
        X = X.reindex(columns=master_cols, fill_value=1.0)
        X_red = ipca.transform(X.to_numpy(dtype=np.float32))
        partial_embeddings.append(X_red)

    X_all = np.vstack(partial_embeddings)
    print(f"[OK] Incremental PCA complete → {X_all.shape}")

    # === Step 3: 在压缩后的数据上运行 UMAP ===
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
