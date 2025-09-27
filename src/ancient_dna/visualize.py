import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS, Isomap
from umap import UMAP
import time


def project_embeddings(X: pd.DataFrame, method: str = "umap", n_components: int = 2, **kwargs) -> pd.DataFrame:
    """
    将基因型矩阵降维到 2D 或 3D，用于可视化。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param method: 降维方法，可选 "umap" / "tsne" / "mds" / "isomap"。
    :param n_components: 目标维度（2 或 3），默认 2。
    :param kwargs: 传给具体模型的额外参数。
    :return: 投影后的 DataFrame。
    """
    if method == "umap":
        # UMAP 支持 random_state
        model = UMAP(n_components=n_components, **kwargs)
    elif method == "tsne":
        # TSNE 支持 random_state
        model = TSNE(n_components=n_components, **kwargs)
    elif method == "mds":
        # MDS 不支持 random_state，移除它
        kwargs.pop("random_state", None)
        model = MDS(n_components=n_components, **kwargs)
    elif method == "isomap":
        # Isomap 不支持 random_state，移除它
        kwargs.pop("random_state", None)
        model = Isomap(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"未知方法: {method}")

    coords = model.fit_transform(X.fillna(0))
    cols = [f"Dim{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, columns=cols)



def plot_embedding(
    df: pd.DataFrame,
    labels: pd.Series | None = None,
    title: str = "Projection",
    save_path: str | None = None,
    figsize: tuple = (8, 6),
    ax=None
) -> None:
    """
    绘制投影结果（支持 2D），可选择保存或画到指定子图。

    :param df: 投影后的 DataFrame，至少包含两列 ["Dim1", "Dim2"]。
    :param labels: 分类标签 (pd.Series)，如 haplogroup，用于着色。可选。
    :param title: 图标题。
    :param save_path: 保存路径（如 "plot.png" 或 "plot.pdf"）。若为 None，则只显示不保存。
    :param figsize: 图像大小，默认 (8, 6)。
    :param ax: matplotlib Axes 对象。如果提供，则绘制到该子图上。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        ax.scatter(df["Dim1"], df["Dim2"],
                   c=pd.Categorical(labels).codes,
                   cmap="tab10", s=30, alpha=0.7)
    else:
        ax.scatter(df["Dim1"], df["Dim2"], s=30, alpha=0.7)

    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_title(title)

    if save_path and ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] 图像已保存到 {save_path}")
    elif ax is None:
        plt.show()

def plot_multiple_embeddings(X, meta, methods, label_col=None,
                             save_path=None, random_state=42, figsize=(16, 12)):
    """
    批量运行多种投影方法并画在同一张图上，同时打印每种方法的耗时。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param meta: 样本注释表 (pd.DataFrame)，包含分类标签等。
    :param methods: 降维方法列表，例如 ["umap", "tsne", "mds", "isomap"]。
    :param label_col: meta 中用于着色的列名（如 "haplogroup"）。
    :param save_path: 若提供，则将整张图保存到文件（PNG/PDF 等）。
    :param random_state: 随机种子（传递给支持的降维方法）。
    :param figsize: 整张图的大小，默认 (16, 12)。
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # 获取标签（如果指定了 label_col）
    labels = meta[label_col] if (label_col and label_col in meta.columns) else None

    for i, method in enumerate(methods):
        # 计时
        start = time.time()
        proj = project_embeddings(X, method=method, n_components=2, random_state=random_state)
        elapsed = time.time() - start
        print(f"{method.upper()} finished in {elapsed:.2f} s")

        # 绘制到子图
        plot_embedding(proj, labels=labels, title=f"{method.upper()} Projection", ax=axes[i])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] 投影对比图已保存到 {save_path}")
    plt.close(fig)   # 防止外部弹窗