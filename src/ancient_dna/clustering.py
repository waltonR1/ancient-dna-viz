"""
clustering.py
-----------------
聚类分析模块（Clustering Layer）
Clustering and group analysis module.

用于在基因型矩阵或降维嵌入空间中执行层次聚类（Hierarchical Clustering），
自动确定聚类数、计算聚类纯度。
Provides hierarchical clustering, automatic cluster-number selection,
cluster purity evaluation.

功能 / Functions:
    - find_optimal_clusters(): 自动搜索最佳聚类数（基于轮廓系数）
      Automatically search for optimal number of clusters using silhouette scores.
    - _run_hierarchical_clustering(): 执行层次聚类核心逻辑并可选绘制树状图。
      Core function for hierarchical clustering with optional dendrogram plot.
    - cluster_high_dimensional(): 在高维 SNP 空间执行聚类。
      Perform clustering in high-dimensional SNP space.
    - cluster_on_embedding(): 在降维结果 (t-SNE / UMAP) 空间聚类。
      Perform clustering on low-dimensional embedding space.
    - compare_clusters_vs_labels(): 对比聚类结果与真实标签一致性。
      Compare clustering assignments against ground-truth labels.

说明 / Description:
    本模块属于分析流程的“聚类层”（Clustering Layer），
    负责从基因型数据或降维结果中提取样本分群结构，
    并结合元数据计算聚类质量与生物学一致性。
    The module serves as the clustering layer of the analysis pipeline,
    discovering group structures in genotype or embedding data,
    and evaluating biological consistency and clustering quality metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def find_optimal_clusters_embedding(X: pd.DataFrame, linkage_method: str = "average", metric: str = "euclidean", cluster_range: range = range(2, 11)) -> tuple[int, list[tuple[int, float]]]:
    """
    自动搜索最佳聚类数（基于 Silhouette Score）
    Automatically determine the optimal number of clusters based on silhouette scores.

    :param X: pd.DataFrame
        输入矩阵（行=样本，列=特征）。
        Input matrix (rows = samples, columns = features).

    :param linkage_method: str
        聚类合并策略（默认 "average"）。
        Linkage method (default "average").

    :param metric: str
        距离度量方式（默认 "euclidean"）。
        Distance metric (default "euclidean").

    :param cluster_range: range
        搜索聚类数范围（默认 2~10）。
        Range of cluster numbers to search (default 2–10).

    :return: (best_k, scores)
        - best_k: 最优聚类数
        - scores: [(k, silhouette)] 轮廓系数列表

    说明 / Notes:
        - 自动遍历 k ∈ [2, 10)，计算平均轮廓系数；
          Iterates over k and computes mean silhouette score.
        - 轮廓系数越高表示聚类越合理。
          Higher score indicates better clustering separation.
        - 最终返回得分最高的聚类数。
          Returns the k with highest silhouette score.
    """
    scores = []
    print(f"[INFO] Searching optimal number of clusters ({cluster_range.start}–{cluster_range.stop - 1}) ...")

    for k in tqdm(cluster_range, desc="[CLUSTER] Searching optimal k"):
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
    return best_k, scores


def _run_hierarchical_clustering(X: pd.DataFrame, n_clusters: int = 5, linkage_method: str = "ward", metric: str = "euclidean", plot: bool = True, show_levels: int = 5, compute_silhouette: bool = True) -> tuple[pd.Series, float | None]:
    """
    执行层次聚类（Hierarchical Clustering）
    Perform hierarchical clustering with optional dendrogram visualization.

    :param X: pd.DataFrame
        输入矩阵。Input data matrix.

    :param n_clusters: int, default=5
        聚类数。Number of clusters.

    :param linkage_method: str, default="ward"
        合并策略。Linkage strategy ("ward", "average", etc.).

    :param metric: str, default="euclidean"
        距离度量。Distance metric.

    :param plot: bool, default=True
        是否绘制树状图。Whether to plot dendrogram.

    :param show_levels: int, default=5
        树状图显示层级。Levels to display in dendrogram.

    :param compute_silhouette: bool, default=True
        是否计算轮廓系数。Compute silhouette score or not.

    :return: tuple(pandas.Series, float | None)
        聚类标签与平均轮廓系数。Cluster labels and optional silhouette score.

    说明 / Notes:
        - linkage() 用于生成层次聚类树结构矩阵。
          linkage() computes the hierarchical tree structure.
        - dendrogram() 绘制层次聚类可视化。
          dendrogram() visualizes sample merging.
        - AgglomerativeClustering 执行最终聚类分配。
          AgglomerativeClustering performs final cluster assignment.
        - silhouette_score 可用于聚类质量评估。
          silhouette_score evaluates cluster separation quality.
    """
    print(f"[INFO] Hierarchical clustering — linkage={linkage_method}, metric={metric}")

    if linkage_method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage requires euclidean distance.")

    if isinstance(X, pd.DataFrame):
        X_np = X.to_numpy(dtype=np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)
    Z = linkage(X_np, method=linkage_method, metric=metric)

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


def cluster_high_dimensional(X_imputed: pd.DataFrame, meta: pd.DataFrame, n_clusters: int = 5):
    """
    在高维 SNP 空间执行层次聚类
    Perform hierarchical clustering directly in high-dimensional SNP space.

    :param X_imputed: pd.DataFrame
        已填补缺失值的基因型矩阵。Imputed SNP matrix.

    :param meta: pd.DataFrame
        样本元数据表。Sample metadata table.

    :param n_clusters: int, default=5
        聚类数。Number of clusters.

    :return: pd.DataFrame
        含聚类结果的元数据表。Metadata with cluster assignments.

    说明 / Notes:
        - 直接在原始高维空间执行聚类。
          Performs clustering on full-dimensional genotype data.
        - 返回的 meta 表新增 “cluster” 列。
          Adds new “cluster” column to metadata table.
        - 可用于与地理/种群标签对比分析。
          Useful for comparing against population or regional labels.
    """
    if X_imputed.shape[0] > 2000:
        print("[WARN] High-dimensional clustering may be very slow for large datasets.")
    labels, score = _run_hierarchical_clustering(X_imputed, n_clusters=n_clusters, plot=False)
    meta["cluster"] = labels.values
    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"[OK] High-dimensional clustering complete — clusters={n_clusters}, silhouette={score_str}")
    return meta


def cluster_on_embedding(embedding_df: pd.DataFrame, meta: pd.DataFrame, n_clusters: int = 5):
    """
    在降维空间 (t-SNE / UMAP) 执行聚类
    Perform clustering in low-dimensional embedding space.

    :param embedding_df: pd.DataFrame
        降维结果 (含 Dim1, Dim2)。Embedding coordinates.

    :param meta: pd.DataFrame
        样本元数据。Sample metadata.

    :param n_clusters: int, default=5
        聚类数。Number of clusters.

    :return: pd.DataFrame
        更新后的 meta 表，包含 “cluster”。
        Updated metadata table including “cluster”

    说明 / Notes:
        - 在降维结果基础上执行层次聚类。
          Performs clustering on 2D/3D embeddings.
        - 计算并打印平均轮廓系数。
          Computes and prints average silhouette score.
        - 输出与原 meta 表顺序一致。
          Output order matches original metadata.
    """
    meta_aligned = meta.loc[embedding_df.index].copy()
    labels, score = _run_hierarchical_clustering(embedding_df, n_clusters=n_clusters, linkage_method="ward", metric="euclidean", plot=False)
    meta_aligned["cluster"] = labels
    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"[OK] Embedding-space clustering complete — clusters={n_clusters}, silhouette={score_str}")
    return meta_aligned


def compare_clusters_vs_labels(meta: pd.DataFrame, cluster_col: str = "cluster_2D", label_col: str = "World Zone") -> pd.DataFrame:
    """
    聚类结果与真实标签对比分析
    Compare clustering assignments with ground-truth labels.

    统计每个聚类簇中主标签（Dominant Label）及其纯度（Dominant %），
    评估聚类与真实分类的一致性。
    Summarizes cluster composition and dominant label purity
    to assess consistency with known classes.

    :param meta: pd.DataFrame
        样本元数据（包含 cluster_col 与 label_col）。
        Metadata including both cluster and label columns.

    :param cluster_col: str
        聚类结果列名（默认 "cluster_2D"）。
        Column name of cluster assignments.

    :param label_col: str
        真实标签列名（默认 "World Zone"）。
        Column name of ground-truth labels.

    :return: pd.DataFrame
        每簇组成统计表，包含主标签与纯度。
        Cluster composition table with dominant label and purity.

    说明 / Notes:
        - 通过交叉统计计算每簇主标签。
          Cross-tabulates cluster vs. label counts.
        - 纯度表示主标签样本比例。
          Purity represents the proportion of dominant label samples.
        - 可用于验证聚类的语义合理性。
          Useful for validating semantic or biological consistency of clusters.
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
