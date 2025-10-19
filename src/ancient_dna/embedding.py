import pandas as pd
from sklearn.manifold import TSNE, MDS, Isomap
from umap import UMAP

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