import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS, Isomap
from umap import UMAP
from pathlib import Path


# ==== 降维算法实现 ====

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


# ==== 统一调度接口 ====

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
    method = method.lower()

    if method == "umap":
        return _compute_umap(X, n_components=n_components, **kwargs)

    if method == "tsne":
        return _compute_tsne(X, n_components=n_components, **kwargs)

    if method == "mds":
        return _compute_mds(X, n_components=n_components, **kwargs)

    if method == "isomap":
        return _compute_isomap(X, n_components=n_components, **kwargs)

    raise ValueError(f"未知降维方法: {method}")


# ==== 可视化 ====

def plot_embedding(
    df: pd.DataFrame,
    labels: pd.Series | None = None,
    title: str = "Projection",
    save_path: str | Path | None = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    绘制降维结果（支持 2D）。

    :param df: 投影后的 DataFrame，至少包含 ["Dim1", "Dim2"]。
    :param labels: 分类标签 (pd.Series)，如 haplogroup，用于着色。可选。
    :param title: 图标题。
    :param save_path: 保存路径（如 "plot.png"）。若为 None，则仅显示。
    :param figsize: 图像大小，默认 (8, 6)。
    说明:
        - 每次调用自动创建新图；
        - 若提供 save_path，则保存为文件；
        - 若未提供 save_path，则直接显示图像。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(df["Dim1"], df["Dim2"],
                             c=pd.Categorical(labels).codes,
                             cmap="tab10", s=30, alpha=0.7)
        handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
        ax.legend(handles, pd.Categorical(labels).categories, title="Y Haplogroup")
    else:
        ax.scatter(df["Dim1"], df["Dim2"], s=30, alpha=0.7)

    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] 图像已保存到 {save_path}")
    else:
        plt.show()

    plt.close(fig)
