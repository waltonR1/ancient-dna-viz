import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS, Isomap
import umap


def project_embeddings(X: pd.DataFrame, method: str = "umap", n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    将基因型矩阵降维到 2D 或 3D，用于可视化。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param method: 降维方法，可选 "umap" / "tsne" / "mds" / "isomap"。
    :param n_components: 目标维度（2 或 3），默认 2。
    :param kwargs: 传给具体模型的额外参数。
    :return: 投影后的 DataFrame，列名为 ["Dim1", "Dim2"] 或 ["Dim1", "Dim2", "Dim3"]。
    """
    if method == "umap":
        model = umap.UMAP(n_components=n_components, **kwargs)
    elif method == "tsne":
        model = TSNE(n_components=n_components, **kwargs)
    elif method == "mds":
        model = MDS(n_components=n_components, **kwargs)
    elif method == "isomap":
        model = Isomap(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"未知方法: {method}")

    coords = model.fit_transform(X.fillna(0))
    cols = [f"Dim{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, columns=cols)


def plot_embedding(df: pd.DataFrame, labels: pd.Series | None = None, title: str = "Projection") -> None:
    """
    绘制 2D 投影结果。

    :param df: 投影后的 DataFrame，至少包含两列 ["Dim1", "Dim2"]。
    :param labels: 分类标签 (pd.Series)，如 haplogroup，用于着色。可选。
    :param title: 图标题。
    """
    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(df["Dim1"], df["Dim2"], c=pd.Categorical(labels).codes, cmap="tab10", s=30, alpha=0.7)
    else:
        plt.scatter(df["Dim1"], df["Dim2"], s=30, alpha=0.7)

    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.title(title)
    plt.show()
