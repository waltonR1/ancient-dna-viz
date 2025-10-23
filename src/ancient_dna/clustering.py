"""
clustering.py
-----------------
层次聚类（Hierarchical Clustering）模块
用于基因型矩阵的结构分析与可视化。

功能：
    - find_optimal_clusters(): 自动搜索最佳聚类数（基于轮廓系数）
    - run_hierarchical_clustering(): 通用层次聚类核心逻辑
    - cluster_highdimensional(): 在高维 SNP 矩阵上执行层次聚类
    - cluster_on_embedding(): 在降维 (t-SNE/UMAP) 空间执行层次聚类
    - plot_cluster_on_embedding(): 聚类结果叠加可视化（含主标签显示）
    - compare_clusters_vs_labels(): 聚类结果与真实标签的一致性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# =========================================================
#  自动确定最佳聚类数
# =========================================================
def find_optimal_clusters(
    X: pd.DataFrame,
    linkage_method: str = "ward",
    metric: str = "euclidean",
    cluster_range: range = range(2, 11),
    plot: bool = True
) -> int:
    """
    自动搜索最佳聚类数（基于 Silhouette Score）。

    :param X: 输入矩阵 (pd.DataFrame)，每行表示样本，每列为特征。
    :param linkage_method: 聚类合并策略（默认 "ward"，即最小方差法）。
    :param metric: 距离度量方式（默认 "euclidean"）。
    :param cluster_range: 聚类数搜索范围（默认 2~10）。
    :param plot: 是否绘制轮廓系数趋势图（默认 True）。
    :return: 最佳聚类数 (int)。

    说明：
        - 自动遍历 k ∈ [2, ..., 10)，计算每个 k 的平均轮廓系数；
        - 轮廓系数越高，表示聚类越合理；
        - 最终返回得分最高的 k。
    """
    scores = []
    print(f"[INFO] Searching optimal number of clusters ({cluster_range.start}–{cluster_range.stop - 1}) ...")

    for k in cluster_range:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, metric=metric)
        labels = model.fit_predict(X)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            scores.append((k, score))
            print(f"  - k={k:<2d} → silhouette={score:.3f}")
        else:
            scores.append((k, -1))  # 无效分簇（全部归为一类）

    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"[OK] Optimal cluster number: {best_k} (silhouette={best_score:.3f})")

    if plot:
        ks, vals = zip(*scores)
        plt.figure(figsize=(7, 4))
        plt.plot(ks, vals, marker="o", lw=2)
        plt.title("Silhouette Score by Cluster Number")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return best_k


# =========================================================
#  通用层次聚类核心逻辑
# =========================================================
def run_hierarchical_clustering(
    X: pd.DataFrame,
    n_clusters: int = 5,
    linkage_method: str = "ward",
    metric: str = "euclidean",
    plot: bool = True,
    show_levels: int = 5,
    compute_silhouette: bool = True,
) -> tuple[pd.Series, float | None]:
    """
    执行层次聚类（Hierarchical Clustering），适用于高维或降维数据。

    :param X: 输入矩阵 (pd.DataFrame)，行=样本，列=特征。
    :param n_clusters: 目标聚类数（默认 5）。
    :param linkage_method: 合并策略（默认 "ward"）。
    :param metric: 距离度量方式（默认 "euclidean"）。
    :param plot: 是否绘制层次聚类树状图（默认 True）。
    :param show_levels: 树状图显示的层级深度（默认 5）。
    :param compute_silhouette: 是否计算轮廓系数（默认 True）。
    :return: (labels, silhouette)
             - labels: 每个样本对应的聚类标签 (pd.Series)
             - silhouette: 聚类的平均轮廓系数（float 或 None）

    说明：
        - linkage() 用于计算层次聚类矩阵；
        - dendrogram() 可视化样本合并过程；
        - AgglomerativeClustering 执行最终分群；
        - 若 compute_silhouette=True，则返回聚类质量指标。
    """
    print(f"[INFO] Hierarchical clustering — linkage={linkage_method}, metric={metric}")
    Z = linkage(X, method=linkage_method, metric=metric)

    if plot:
        plt.figure(figsize=(14, 6))
        plt.title(f"Hierarchical Clustering Dendrogram ({linkage_method})")
        dendrogram(Z, truncate_mode="level", p=show_levels)
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric=metric)
    labels = model.fit_predict(X)

    score = None
    if compute_silhouette and len(np.unique(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"[INFO] Silhouette score: {score:.3f}")

    return pd.Series(labels, name="cluster", index=X.index), score


# =========================================================
#  高维 SNP 聚类
# =========================================================
def cluster_highdimensional(X_imputed: pd.DataFrame, meta: pd.DataFrame, n_clusters: int = 5):
    """
    在高维 SNP 空间上执行层次聚类。

    :param X_imputed: 已填补缺失值的 SNP 矩阵。
    :param meta: 样本元数据表（包含标签信息）。
    :param n_clusters: 目标聚类数（默认 5）。
    :return: 含聚类结果的 meta 表。
    """
    labels, score = run_hierarchical_clustering(X_imputed, n_clusters=n_clusters, plot=False)
    meta["cluster"] = labels.values
    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"[OK] High-dimensional clustering complete — clusters={n_clusters}, silhouette={score_str}")
    return meta


# =========================================================
#  降维空间聚类
# =========================================================
def cluster_on_embedding(embedding_df: pd.DataFrame, meta: pd.DataFrame, n_clusters: int = 5):
    """
    在降维空间 (t-SNE / UMAP) 上执行层次聚类。

    :param embedding_df: 降维结果 (包含 Dim1, Dim2)。
    :param meta: 样本元数据表。
    :param n_clusters: 聚类数（默认 5）。
    :return: 更新后的 meta（含 cluster_2D 列）。
    """
    labels, score = run_hierarchical_clustering(embedding_df, n_clusters=n_clusters, plot=False)
    meta["cluster_2D"] = labels.values
    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"[OK] Embedding-space clustering complete — clusters={n_clusters}, silhouette={score_str}")
    return meta


def plot_cluster_on_embedding(
    embedding_df: pd.DataFrame,
    labels: pd.Series,
    meta: pd.DataFrame | None = None,
    label_col: str = "World Zone",
    title: str = "Clusters on Embedding Space",
    figsize: tuple = (8, 6)
):
    """
    绘制聚类结果叠加在降维结果（t-SNE / UMAP）上，
    并在每个簇中心标注主要标签（Dominant Label）及纯度（Dominant %）。

    :param embedding_df: 降维结果 (包含 Dim1, Dim2)。
    :param labels: 聚类标签序列 (pd.Series)。
    :param meta: 样本元数据表（包含真实标签列）。
    :param label_col: 真实标签列名（默认 "World Zone"）。
    :param title: 图标题。
    :param figsize: 图像大小 (宽, 高)。
    :return: None

    说明：
        - 点颜色代表聚类簇；
        - 若 meta 与 label_col 提供，则在每簇中心显示：
              Europe(W) – 98.4%
        - 其中百分比为主标签在该簇中的占比（聚类纯度）。
    """
    if "Dim1" not in embedding_df.columns or "Dim2" not in embedding_df.columns:
        raise ValueError("embedding_df must contain columns 'Dim1' and 'Dim2'.")

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        embedding_df["Dim1"],
        embedding_df["Dim2"],
        c=labels,
        cmap="tab20",
        s=40,
        alpha=0.8,
        edgecolor="none"
    )
    plt.title(title)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.colorbar(scatter, label="Cluster ID")

    # ====== 在簇中心标注 dominant label + purity ======
    if meta is not None and label_col in meta.columns:
        meta = meta.copy()
        meta["cluster_2D"] = labels.values

        # 计算簇中心坐标
        group_centers = (
            embedding_df.assign(cluster=labels)
            .groupby("cluster")[["Dim1", "Dim2"]]
            .mean()
        )

        # 计算每簇主标签及纯度
        cluster_stats = (
            meta.groupby(["cluster_2D", label_col])
            .size()
            .unstack(fill_value=0)
        )
        cluster_stats["Total"] = cluster_stats.sum(axis=1)
        cluster_stats["Dominant Label"] = cluster_stats.drop(columns="Total").idxmax(axis=1)
        cluster_stats["Dominant %"] = (
            cluster_stats.apply(lambda r: r[r["Dominant Label"]] / r["Total"], axis=1) * 100
        ).round(1)

        # 在每簇中心绘制标签与纯度
        for cluster_id, (x, y) in group_centers.iterrows():
            if cluster_id in cluster_stats.index:
                label = cluster_stats.loc[cluster_id, "Dominant Label"]
                purity = cluster_stats.loc[cluster_id, "Dominant %"]
                text = f"{label} – {purity:.1f}%"
                plt.text(
                    x, y, text,
                    fontsize=10, weight="bold",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
                )

    plt.tight_layout()
    plt.show()



# =========================================================
#  聚类结果 vs 真实标签对比
# =========================================================
def compare_clusters_vs_labels(
    meta: pd.DataFrame,
    cluster_col: str = "cluster_2D",
    label_col: str = "World Zone"
) -> pd.DataFrame:
    """
    对比聚类结果与真实标签（如 World Zone）之间的一致性，
    输出每个聚类的主标签及纯度统计。

    :param meta: 样本元数据表（需包含 cluster_col 与 label_col）。
    :param cluster_col: 聚类结果列名（默认 "cluster_2D"）。
    :param label_col: 真实标签列名（默认 "World Zone"）。
    :return: 聚类组成统计表 (pd.DataFrame)，含每簇的主要标签及纯度。

    说明：
        - 按 cluster_col 与 label_col 交叉统计；
        - Dominant Label 表示该簇内出现最多的真实标签；
        - Dominant % 表示主标签样本占该簇总样本的比例（聚类纯度）。
    """
    if cluster_col not in meta.columns or label_col not in meta.columns:
        raise ValueError(f"Missing column: {cluster_col} or {label_col}")

    summary = (
        meta.groupby([cluster_col, label_col])
        .size()
        .unstack(fill_value=0)
        .astype(int)
    )

    summary["Total"] = summary.sum(axis=1)
    summary["Dominant Label"] = summary.drop(columns="Total").idxmax(axis=1)

    # pandas >= 2.0 兼容写法
    summary["Dominant %"] = (
        summary.apply(lambda r: r[r["Dominant Label"]] / r["Total"], axis=1) * 100
    ).round(1)

    print("\n[INFO] Cluster composition summary:")
    print(summary)

    return summary
