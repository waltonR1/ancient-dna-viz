"""
clustering.py
=============

聚类分析模块（Clustering Layer）
Clustering and group analysis module.

本模块用于在基因型矩阵或降维嵌入空间中执行层次聚类
（Hierarchical Clustering），支持自动确定聚类数、
计算聚类纯度等分析任务。

This module provides hierarchical clustering utilities for
genotype matrices and low-dimensional embeddings, including
automatic cluster-number selection and cluster purity evaluation.

Functions
---------
find_optimal_clusters_embedding
    自动搜索最佳聚类数（基于轮廓系数）。
    Automatically search for the optimal number of clusters
    using silhouette scores.

_run_hierarchical_clustering
    执行层次聚类的核心逻辑，并可选绘制树状图。
    Core function for hierarchical clustering with optional
    dendrogram visualization.

cluster_high_dimensional
    在高维 SNP 空间中执行聚类。
    Perform clustering in high-dimensional SNP space.

cluster_on_embedding
    在降维结果（t-SNE / UMAP）空间中执行聚类。
    Perform clustering on low-dimensional embedding space.

compare_clusters_vs_labels
    对比聚类结果与真实标签的一致性，计算聚类纯度。
    Compare clustering assignments against ground-truth labels
    and evaluate cluster purity.

Description
-----------
本模块属于分析流程中的“聚类层（Clustering Layer）”，
负责从基因型数据或降维结果中提取样本的分群结构，
并结合元数据评估聚类质量及其生物学一致性。

The module serves as the clustering layer of the analysis pipeline,
discovering group structures in genotype or embedding data and
evaluating clustering quality and biological consistency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def find_optimal_clusters_embedding(
    X: pd.DataFrame,
    linkage_method: str = "average",
    metric: str = "euclidean",
    cluster_range: range = range(2, 11),
) -> tuple[int, list[tuple[int, float]]]:
    """
    自动搜索最佳聚类数（基于 Silhouette Score）

    Automatically determine the optimal number of clusters using silhouette scores.

    本函数在给定的聚类数范围内遍历不同的 k 值，
    对每一个 k 执行层次聚类（Agglomerative Clustering），
    并计算对应的平均轮廓系数（Silhouette Score），
    最终返回轮廓系数最高的聚类数。

    The function iterates over a range of cluster numbers and performs
    hierarchical clustering for each value of ``k``. The optimal
    number of clusters is selected based on the highest average
    silhouette score.

    Parameters
    ----------
    X : pandas.DataFrame
        输入特征矩阵，行表示样本，列表示特征。

        Input feature matrix where rows correspond to samples and
        columns correspond to features.

    linkage_method : str, default="average"
        层次聚类的合并策略（如 ``"ward"``、``"average"``、``"complete"``）。

        Linkage strategy used in hierarchical clustering.

    metric : str, default="euclidean"
        距离度量方式。

        Distance metric used to compute inter-sample distances.

    cluster_range : range, default=range(2, 11)
        搜索的聚类数范围（默认 2–10）。

        Range of cluster numbers to evaluate.

    Returns
    -------
    best_k : int
        最优聚类数（对应最高的 Silhouette Score）。

        Optimal number of clusters corresponding to the highest
        silhouette score.

    scores : list of tuple (int, float)
        每个 k 对应的 ``(k, silhouette_score)`` 列表。

        List of ``(k, silhouette_score)`` pairs for all tested
        cluster numbers.

    Notes
    -----
    - 轮廓系数越高，表示类内更紧凑、类间更分离。

      Higher silhouette scores indicate better-defined clusters.

    - 当聚类结果退化为单一簇时，其轮廓系数记为 ``-1``。

      Degenerate clustering solutions are assigned a score of ``-1``.

    - 本函数假设输入特征已完成必要的预处理（如标准化）。

      The function assumes that input features have already been
      preprocessed if necessary.
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


def _run_hierarchical_clustering(
        X: pd.DataFrame,
        n_clusters: int = 5,
        linkage_method: str = "ward",
        metric: str = "euclidean",
        plot: bool = True,
        show_levels: int = 5,
        compute_silhouette: bool = True,
) -> tuple[pd.Series, float | None]:
    """
    执行一次完整的层次聚类流程（内部通用函数）

    Perform a complete hierarchical clustering procedure (internal utility).

    本函数封装了层次聚类的核心流程，包括：
    1）构建层次聚类树结构（用于可视化）
    2）可选绘制树状图（dendrogram）
    3）使用 AgglomerativeClustering 生成最终聚类标签
    4）可选计算平均轮廓系数（Silhouette Score）

    This function encapsulates the core steps of hierarchical clustering,
    including hierarchical tree construction, optional dendrogram
    visualization, final cluster assignment, and optional silhouette
    score computation.

    Parameters
    ----------
    X : pandas.DataFrame
        输入特征矩阵，行表示样本，列表示特征。

        Input feature matrix where rows correspond to samples and
        columns correspond to features.

    n_clusters : int, default=5
        最终切分得到的聚类数。

        Number of clusters to form.

    linkage_method : str, default="ward"
        层次聚类的合并策略（如 ``"ward"``、``"average"``）。

        Linkage strategy used in hierarchical clustering.

    metric : str, default="euclidean"
        距离度量方式。

        Distance metric used to compute inter-sample distances.

    plot : bool, default=True
        是否绘制层次聚类树状图（dendrogram）。

        Whether to plot the dendrogram.

    show_levels : int, default=5
        树状图中显示的层级深度（用于截断显示）。

        Number of hierarchical levels to display in the dendrogram.

    compute_silhouette : bool, default=True
        是否计算并返回平均轮廓系数。

        Whether to compute and return the silhouette score.

    Returns
    -------
    labels : pandas.Series
        每个样本对应的聚类标签，索引与输入数据一致。

        Cluster labels indexed by sample.

    score : float or None
        平均轮廓系数；若未计算或聚类退化则为 ``None``。

        Mean silhouette score if computed, otherwise ``None``.

    Notes
    -----
    - 本函数同时使用 ``scipy.cluster.hierarchy.linkage`` 和
      ``sklearn.cluster.AgglomerativeClustering``：

      The function uses both ``scipy.cluster.hierarchy.linkage`` and
      ``sklearn.cluster.AgglomerativeClustering``.

    - ``linkage`` 仅用于构建层次结构和绘制树状图，
      并不决定最终的聚类标签。

      ``linkage`` is used only for hierarchical structure construction
      and visualization, not for final cluster assignment.

    - 当 ``linkage_method="ward"`` 时，仅支持欧氏距离，
      否则会抛出异常。

      Ward linkage requires Euclidean distance; other metrics
      will raise an error.

    - 当聚类结果退化为单一簇时，不计算轮廓系数。

      Silhouette score is not computed for degenerate clustering
      solutions.
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


def cluster_high_dimensional(
        X_imputed: pd.DataFrame,
        meta: pd.DataFrame,
        n_clusters: int = 5,
):
    """
    在高维 SNP 特征空间中执行层次聚类

    Perform hierarchical clustering directly in high-dimensional SNP space.

    本函数直接在原始的高维基因型特征空间（如 SNP 矩阵）中
    执行层次聚类，并将聚类结果写入样本元数据表中。

    The function performs hierarchical clustering directly on the
    full-dimensional genotype feature space and appends the resulting
    cluster assignments to the metadata table.

    Parameters
    ----------
    X_imputed : pandas.DataFrame
        已完成缺失值填补的基因型矩阵（样本 × SNP）。

        Imputed genotype matrix (samples × SNPs).

    meta : pandas.DataFrame
        与样本一一对应的元数据表。

        Sample metadata table aligned with the genotype matrix.

    n_clusters : int, default=5
        聚类数。

        Number of clusters to form.

    Returns
    -------
    pandas.DataFrame
        在原元数据表基础上新增 ``"cluster"`` 列，
        表示每个样本所属的聚类编号。

        Metadata table with an additional ``"cluster"`` column
        containing cluster assignments.

    Notes
    -----
    - 本函数在高维空间中计算样本间距离，
      当样本数量较大时，计算开销可能较高。

      Distance computation in high-dimensional space can be
      computationally expensive for large datasets.

    - 当样本数量超过一定规模时，会给出性能警告。

      A warning is issued when the dataset size exceeds
      a practical threshold.

    - 该函数主要用于验证性分析或与降维聚类结果进行对照。

      This function is mainly intended for validation or
      comparison with clustering results in embedding space.
    """

    if X_imputed.shape[0] > 2000:
        print("[WARN] High-dimensional clustering may be very slow for large datasets.")
    labels, score = _run_hierarchical_clustering(X_imputed, n_clusters=n_clusters, plot=False)
    meta["cluster"] = labels.values
    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"[OK] High-dimensional clustering complete — clusters={n_clusters}, silhouette={score_str}")
    return meta


def cluster_on_embedding(
        embedding_df: pd.DataFrame,
        meta: pd.DataFrame,
        n_clusters: int = 5,
):
    """
    在降维嵌入空间中执行层次聚类（UMAP / t-SNE 等）

    Perform hierarchical clustering in low-dimensional embedding space.

    本函数在降维后的嵌入坐标空间（如 UMAP、t-SNE 或 PCA）中
    执行层次聚类，并将聚类结果写入样本元数据表。

    The function performs hierarchical clustering in a low-dimensional
    embedding space (e.g. UMAP, t-SNE, or PCA) and appends the resulting
    cluster assignments to the metadata table.

    Parameters
    ----------
    embedding_df : pandas.DataFrame
        降维后的嵌入坐标矩阵（如 Dim1, Dim2, [Dim3]）。

        Low-dimensional embedding coordinates (e.g. Dim1, Dim2, [Dim3]).

    meta : pandas.DataFrame
        样本元数据表，与嵌入坐标按索引对齐。

        Sample metadata table aligned with the embedding indices.

    n_clusters : int, default=5
        聚类数。

        Number of clusters to form.

    Returns
    -------
    pandas.DataFrame
        对齐后的元数据表，并新增 ``"cluster"`` 列，
        表示每个样本在嵌入空间中的聚类编号。

        Metadata table aligned to the embedding with an additional
        ``"cluster"`` column.

    Notes
    -----
    - 聚类仅基于降维后的嵌入坐标，
      不再使用原始高维基因型特征。

      Clustering is performed only on the embedding coordinates,
      not on the original high-dimensional genotype data.

    - 元数据表会根据嵌入矩阵的索引进行对齐。

      The metadata table is aligned to the embedding indices
      before cluster assignment.

    - 该函数通常用于可视化分析或与已知标签进行对照。

      This function is commonly used for visualization and
      interpretative analysis.
    """

    meta_aligned = meta.loc[embedding_df.index].copy()
    labels, score = _run_hierarchical_clustering(embedding_df, n_clusters=n_clusters, linkage_method="ward", metric="euclidean", plot=False)
    meta_aligned["cluster"] = labels
    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"[OK] Embedding-space clustering complete — clusters={n_clusters}, silhouette={score_str}")
    return meta_aligned


def compare_clusters_vs_labels(
        meta: pd.DataFrame,
        cluster_col: str = "cluster",
        label_col: str = "World Zone",
) -> pd.DataFrame:
    """
    聚类结果与真实标签的一致性分析

    Compare clustering assignments with ground-truth labels.

    本函数统计每个聚类簇中不同真实标签的样本数量，
    识别每个簇的主导标签（Dominant Label），
    并计算其所占比例（聚类纯度）。

    The function summarizes the composition of each cluster
    with respect to known labels, identifies the dominant
    label per cluster, and computes cluster purity.

    Parameters
    ----------
    meta : pandas.DataFrame
        包含聚类结果列和真实标签列的样本元数据表。

        Sample metadata table containing both cluster and
        ground-truth label columns.

    cluster_col : str, default="cluster"
        聚类结果所在的列名。

        Column name containing cluster assignments.

    label_col : str, default="World Zone"
        真实标签所在的列名（如地理区域或种群标签）。

        Column name containing ground-truth labels.

    Returns
    -------
    pandas.DataFrame
        聚类组成统计表，每一行对应一个聚类簇，包含：

        - 各真实标签在该簇中的样本数量
        - ``Total``：该簇的样本总数
        - ``Dominant Label``：样本数最多的真实标签
        - ``Dominant %``：主导标签所占比例（百分比）

        A cluster composition summary table including:
        counts per label, total sample size, dominant label,
        and dominant label percentage.

    Notes
    -----
    - 聚类纯度定义为主导标签样本数除以该簇样本总数。

      Cluster purity is defined as the proportion of samples
      belonging to the dominant label.

    - 该指标为描述性统计，不考虑标签之间的语义或距离关系。

      This metric is descriptive and does not account for
      semantic or hierarchical relationships between labels.

    - 该函数适用于分析聚类结果的可解释性，
      而非作为聚类算法的优化目标。

      This function is intended for interpretability analysis,
      not as an optimization criterion.
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
