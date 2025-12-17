"""
visualize.py
-----------------
可视化模块（Visualization Layer）
Visualization module for plotting embeddings and missingness patterns.

用于绘制降维嵌入结果、样本分布、缺失数据及聚类结构可视化图。
提供从中小规模数据到大矩阵的智能绘制方案，
支持颜色映射、图例布局与结果导出。
Used to visualize dimensionality reduction embedding results, sample distribution, missing data, and cluster structure.
Supports both fine-grained and aggregated plotting modes with customizable color maps,
legend layouts, and export options.

功能 / Functions:
    - plot_embedding(): 绘制二维降维投影结果并可选分组上色；
      Plot 2D embedding projection with optional categorical coloring.
    - plot_missing_values(): 智能绘制缺失值分布图（小矩阵像素图 / 大矩阵聚合图）。
      Smart visualization of missing-value patterns (pixel-based or aggregated modes).
    - plot_cluster_on_embedding(): 绘制聚类结果叠加图，并显示主标签与纯度。
      Visualize clusters on embeddings, showing dominant label and purity.
    - plot_silhouette_trend(): 绘制聚类数与平均轮廓系数的关系趋势图。
      Plot the relationship between number of clusters and silhouette score.

说明 / Description:
    本模块作为数据分析的“可视化层”，
    可辅助评估降维结果质量、样本分布一致性与缺失数据结构。
    图像风格遵循出版级别的清晰度要求，默认使用 Matplotlib 原生 API。
    This module serves as the visualization layer of the pipeline,
    helping assess embedding quality, sample clustering, and missing-data structure.
    Figures are designed for publication-level clarity using standard Matplotlib APIs.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines
from pathlib import Path
from matplotlib.colors import ListedColormap
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go

_WORLD_ZONE_CMAPS = {
    "Europe": get_cmap("Blues"),
    "Asia": get_cmap("Greens"),
    "Africa": get_cmap("YlOrBr"),
    "America": get_cmap("Reds"),
    "Oceania": get_cmap("Purples"),
    "Middle East": get_cmap("Oranges"),
}

_WORLD_ZONE_ORDER = [
    "Europe",
    "Asia",
    "Middle East",
    "Africa",
    "America",
    "Oceania",
]

def _categorical_color_mapping(labels: pd.Series, legend_max: int, legend_sort: bool, cmap: str, others_color: tuple):
    labels = labels.astype(str)

    # 原始排序
    if legend_sort:
        ordered = labels.value_counts().index.tolist()
    else:
        ordered = labels.unique().tolist()

    main_classes = ordered[:legend_max]
    other_classes = ordered[legend_max:]

    # 判断语义类型
    sample = pd.Series(main_classes)

    is_world_zone = sample.str.startswith(tuple(_WORLD_ZONE_CMAPS)).mean() > 0.6

    class_to_color = {}
    main_colors = []

    # World Zone 语义映射
    if is_world_zone:
        groups = defaultdict(list)
        for cls in main_classes:
            for cont in _WORLD_ZONE_ORDER:
                if cls.startswith(cont):
                    groups[cont].append(cls)
                    break
            else:
                groups["Other"].append(cls)

        main_classes = []
        for cont in _WORLD_ZONE_ORDER + ["Other"]:
            if cont not in groups:
                continue
            classes = sorted(groups[cont])
            cmap_cont = _WORLD_ZONE_CMAPS.get(cont, get_cmap(cmap))
            shades = np.linspace(0.4, 0.85, len(classes))

            for cls, shade in zip(classes, shades):
                color = cmap_cont(shade)
                class_to_color[cls] = color
                main_classes.append(cls)
                main_colors.append(color)

    # 普通 categorical
    else:
        base_cmap = get_cmap(cmap)
        main_colors = [
            base_cmap(i / max(1, len(main_classes) - 1))
            for i in range(len(main_classes))
        ]
        class_to_color = dict(zip(main_classes, main_colors))

    # ===== Step 3: others =====
    for cls in other_classes:
        class_to_color[cls] = others_color

    return class_to_color, main_classes, other_classes, main_colors


def plot_embedding( df: pd.DataFrame, labels: pd.Series | None = None, title: str = "Projection", save_path: str | Path | None = None, figsize: tuple = (10, 7), legend_pos: str = "right", cmap: str = "tab20", legend_max: int = 20, legend_sort: bool = True, others_color : tuple = (0.7, 0.7, 0.7, 0.5),draw_others: bool = False) -> None:
    """
    绘制 2D / 3D 降维结果
    Plot 2D or 3D embedding projection with optional categorical coloring.

    根据输入的二维降维结果与可选标签进行散点图绘制，
    支持类别排序、颜色映射、图例位置控制及保存导出。
    Plots 2D projection of embedding results with optional categorical coloring,
    supporting label sorting, color mapping, legend positioning, and export.

    :param df: pd.DataFrame
        投影后的 DataFrame，至少包含 ["Dim1", "Dim2"]。
        Projected DataFrame with at least ["Dim1", "Dim2"] columns.

    :param labels: pd.Series | None
        分类标签（用于着色，可选）。
        Categorical labels used for color mapping (optional).

    :param title: str
        图标题。
        Plot title.

    :param save_path: str | Path | None
        保存路径（如 "plot.png"）。若为 None，则直接显示。
        Save path for figure; if None, the plot is shown interactively.

    :param figsize: tuple, default=(10,7)
        图像尺寸。
        Figure size in inches.

    :param legend_pos: str
        图例位置，可选 {"right", "bottom", "top", "inside"}。
        Legend position: {"right", "bottom", "top", "inside"}.

    :param cmap: str, default="tab20"
        颜色映射表名称。
        Name of Matplotlib color map.

    :param legend_max: int, default=20
        图例中最大显示类别数。超过则合并为灰色“others”。
        Maximum number of classes displayed in legend; remaining merged as "others".

    :param legend_sort: bool, default=True
        是否按样本数量排序。
        Whether to sort categories by sample size.

    :param others_color: tuple
        超出 legend 限制的类别所使用的灰色。
        Color for merged "others" classes.

    :param draw_others: bool, default=True
        是否绘制other类的点
        Whether to draw points for "others" classes.

    说明 / Notes:
        - 图中散点与图例颜色保持一致。
          Scatter point colors match legend colors exactly.
        - 当类别超过 legend_max 时，多余类别合并为灰色。
          Classes beyond legend_max are merged into a single gray group.
        - 图例可灵活布局到右侧、底部或顶部。
          Legends can be positioned on the right, bottom, top, or inside plot.
        - 支持直接展示或保存为高分辨率 PNG。
          Supports direct display or high-resolution PNG export.
    """
    # 判断是 2D 还是 3D
    is_3d = "Dim3" in df.columns

    # 初始化 figure
    if is_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # 如果有标签，执行颜色映射
    if labels is not None:
        class_to_color, main_classes, other_classes, main_colors = _categorical_color_mapping(
                labels=labels,
                legend_max=legend_max,
                legend_sort=legend_sort,
                cmap=cmap,
                others_color=others_color,
            )

        # 是否过滤 others
        if not draw_others:
            mask = labels.isin(main_classes)
            df = df[mask]
            labels = labels[mask]

        # 将颜色映射到每一个点
        point_colors = labels.astype(str).map(class_to_color)

        # scatter 绘制
        if is_3d:
            ax.scatter(
                df["Dim1"], df["Dim2"], df["Dim3"],
                c=point_colors, s=18, alpha=0.7
            )
        else:
            ax.scatter(
                df["Dim1"], df["Dim2"],
                c=point_colors, s=20, alpha=0.65
            )

        # legend：如果不画 others，则不显示 "others"
        if draw_others and other_classes:
            legend_classes = main_classes + ["… (others)"]
            legend_colors = main_colors + [others_color]
        else:
            legend_classes = main_classes
            legend_colors = main_colors

        handles = [
            mlines.Line2D([], [], color=legend_colors[i], marker="o",
                          linestyle="None", markersize=6)
            for i in range(len(legend_classes))
        ]

        # 图例位置布局
        if legend_pos == "right":
            loc, bbox, rect = "center left", (1.02, 0.5), (0, 0, 0.85, 1)
        elif legend_pos == "bottom":
            loc, bbox, rect = "upper center", (0.5, -0.15), (0, 0.1, 1, 1)
        elif legend_pos == "top":
            loc, bbox, rect = "lower center", (0.5, 1.15), (0, 0, 1, 0.9)
        elif legend_pos == "inside":
            loc, bbox, rect = "upper right", None, (0, 0, 1, 1)
        else:
            raise ValueError(f"Invalid legend_pos: {legend_pos}")

        ax.legend(handles, legend_classes, title=labels.name,
                  loc=loc, bbox_to_anchor=bbox, frameon=False,
                  fontsize=9, title_fontsize=10)

    else:
        # ====== 无标签情况 ======
        if is_3d:
            ax.scatter(df["Dim1"], df["Dim2"], df["Dim3"],
                       s=20, alpha=0.7, color="gray")
        else:
            ax.scatter(df["Dim1"], df["Dim2"],
                       s=20, alpha=0.7, color="gray")
        rect = [0, 0, 1, 1]

    # ====== Step 5. 坐标轴与标题 ======
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    if is_3d:
        ax.set_zlabel("Dim3")
    ax.set_title(title)

    # 紧凑布局
    if labels is not None:
        plt.tight_layout(rect = rect)
    else:
        plt.tight_layout()

    # ====== Step 6. 保存或显示 ======
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved → {save_path}")
    else:
        plt.show()
        print("[OK} Figure shown interactively.")

    plt.close(fig)


def plot_embedding_interactive(
    df: pd.DataFrame,
    labels: pd.Series,
    dim: int | None = None,
    title: str = "Projection",
    legend_max: int = 20,
    legend_sort: bool = False,
    cmap: str = "tab20",
    others_color: tuple = (0.7, 0.7, 0.7, 0.5),
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Interactive embedding visualization using Plotly.

    Supports:
    - 2D and 3D embedding
    - Interactive legend (click to hide/show categories)
    - Zoom / pan (2D)
    - Rotate / zoom (3D)
    - Reuses the same categorical color mapping as matplotlib version

    Parameters
    ----------
    df : pd.DataFrame
        Embedding dataframe, must contain Dim1, Dim2 (and Dim3 if dim=3).

    labels : pd.Series
        Categorical labels for coloring.

    dim : int, default=2
        Dimension of embedding (2 or 3).

    title : str
        Plot title.

    legend_max : int
        Maximum number of legend classes.

    legend_sort : bool
        Whether to sort classes (passed to color mapping).

    cmap : str
        Fallback colormap name.

    others_color : tuple
        RGBA color for merged "others".

    show : bool
        Whether to show the figure immediately.

    save_path : str | Path | None
        If provided, save the figure as HTML.
    """
    if dim is None:
        dim = 3 if "Dim3" in df.columns else 2

    labels = labels.astype(str)

    # ===== Step 1: reuse categorical color mapping =====
    class_to_color, main_classes, other_classes, _ = _categorical_color_mapping(
        labels=labels,
        legend_max=legend_max,
        legend_sort=legend_sort,
        cmap=cmap,
        others_color=others_color,
    )

    fig = go.Figure()

    # ===== Step 2: one trace per class (key for interactive legend) =====
    for cls in main_classes:
        mask = labels == cls
        color = class_to_color[cls]

        # Convert RGBA (0–1) to CSS rgba string
        rgba = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"

        if dim == 2:
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "Dim1"],
                    y=df.loc[mask, "Dim2"],
                    mode="markers",
                    name=cls,
                    marker=dict(color=rgba, size=6),
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=df.loc[mask, "Dim1"],
                    y=df.loc[mask, "Dim2"],
                    z=df.loc[mask, "Dim3"],
                    mode="markers",
                    name=cls,
                    marker=dict(color=rgba, size=4),
                )
            )

    # ===== Step 3: others =====
    if other_classes:
        mask = labels.isin(other_classes)
        rgba = f"rgba({int(others_color[0]*255)}, {int(others_color[1]*255)}, {int(others_color[2]*255)}, {others_color[3]})"

        if dim == 2:
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "Dim1"],
                    y=df.loc[mask, "Dim2"],
                    mode="markers",
                    name="… (others)",
                    marker=dict(color=rgba, size=5),
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=df.loc[mask, "Dim1"],
                    y=df.loc[mask, "Dim2"],
                    z=df.loc[mask, "Dim3"],
                    mode="markers",
                    name="… (others)",
                    marker=dict(color=rgba, size=4),
                )
            )

    # ===== Step 4: layout =====
    if dim == 2:
        fig.update_layout(
            title=title,
            xaxis_title="Dim1",
            yaxis_title="Dim2",
            legend=dict(itemsizing="constant"),
        )
    else:
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Dim1",
                yaxis_title="Dim2",
                zaxis_title="Dim3",
            ),
            legend=dict(itemsizing="constant"),
        )

    # ===== Step 5: output =====
    if save_path:
        fig.write_html(str(save_path))
        print(f"[OK] Interactive figure saved → {save_path}")

    if show:
        fig.show()

    return fig


def plot_missing_values(df: pd.DataFrame, save_path: str | Path | None = None, missing_value: int = 3, figsize: tuple = (20, 10), cmap_present: str = "#d95f02", cmap_missing: str = "#ffffff", show_ratio: bool = True, max_pixels: int = 5e7) -> None:
    """
    智能绘制缺失数据可视化图
    Intelligent visualization of missing data patterns.

    根据矩阵规模自动切换绘制模式：
    小矩阵绘制逐像素缺失图，大矩阵绘制缺失率分布直方图。
    Automatically switches between full pixel-plot and aggregated histogram modes
    depending on the dataset size.

    :param df: pd.DataFrame
        基因样本矩阵。
        Genotype matrix.

    :param save_path: str | Path | None
        保存路径（如 "plot.png"）。若为 None，则直接显示。
        Save path for figure; if None, the plot is shown interactively.

    :param missing_value: int, default=3
        缺失值标记符。
        Value used to represent missing entries (default = 3).

    :param figsize: tuple, default=(20,10)
        图像尺寸（宽, 高）。
        Figure size (width, height).

    :param cmap_present: str
        非缺失值颜色（默认橙色 #d95f02）。
        Color for present (non-missing) values.

    :param cmap_missing: str
        缺失值颜色（默认白色 #ffffff）。
        Color for missing entries.

    :param show_ratio: bool
        是否在下方附带每列缺失率柱状图。
        Whether to display per-column missing ratios below the matrix.

    :param max_pixels: int
        当矩阵元素数超过该阈值时自动使用聚合模式。
        Threshold for switching to aggregated histogram mode.

    说明 / Notes:
        - 小矩阵绘制逐像素模式，适合人工检查。
          Small matrices use pixel visualization for manual inspection.
        - 大矩阵自动改用分布图，避免内存溢出与渲染延迟。
          Large matrices are summarized via histograms for efficiency.
        - 可显示或保存为高分辨率 PNG 文件。
          Supports on-screen rendering or PNG export.
        - 缺失率统计可同时反映样本与 SNP 层面数据完整性。
          Missingness distributions reflect both sample- and SNP-level quality.
    """
    n_rows, n_cols = df.shape
    total = n_rows * n_cols

    if total > max_pixels:
        # ======= 大矩阵模式：聚合绘制 =======
        print(f"[INFO] Large matrix detected ({n_rows}×{n_cols} ≈ {total/1e6:.1f}M pixels)")
        print("[INFO] Switching to aggregated missingness summary mode...")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        row_missing = (df == missing_value).mean(axis=1)
        col_missing = (df == missing_value).mean(axis=0)

        ax[0].hist(row_missing, bins=50, color="#4c72b0", alpha=0.8)
        ax[0].set_title("Sample missing rate distribution")
        ax[0].set_xlabel("Missing rate per sample")
        ax[0].set_ylabel("Count")

        ax[1].hist(col_missing.sample(min(len(col_missing), 5000), random_state=42),
                   bins=50, color="#dd8452", alpha=0.8)
        ax[1].set_title("SNP missing rate distribution (sampled)")
        ax[1].set_xlabel("Missing rate per SNP")
        ax[1].set_ylabel("Count")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Aggregated missingness plot saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)
        return

    # ======= 正常模式：原像素绘图 =======
    print("[INFO] Plotting missing value pattern...")
    # Step 1. 创建缺失掩码
    mask = (df == missing_value).to_numpy()

    # Step 2. 画布布局
    if show_ratio:
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

    # Step 3. 绘制缺失矩阵（白 = 缺失）
    ax = axes[0]
    ax.imshow(mask, aspect='auto', cmap=ListedColormap([cmap_present, cmap_missing]), interpolation='none')
    ax.set_title(f"Missing Data Pattern")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Samples")

    # Step 4. 绘制缺失比例（可选）
    if show_ratio:
        ax_ratio = axes[1]
        missing_ratio = (df == missing_value).mean() * 100
        ax_ratio.bar(np.arange(len(df.columns)), missing_ratio, color="#ff7043")
        ax_ratio.set_ylabel("Missing (%)")
        ax_ratio.set_xticks([])
        ax_ratio.set_ylim(0, 100)
        ax_ratio.grid(axis="y", linestyle="--", alpha=0.3)
        ax_ratio.set_title("Missing Ratio per Column")

    # Step 5. 调整布局
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Missing value plot已保存到 {save_path}")
    else:
        print("[OK] Displayed missing value pattern for inspection.")
        plt.show()

    plt.close(fig)


def plot_cluster_on_embedding( embedding_df: pd.DataFrame, labels: pd.Series, meta: pd.DataFrame | None = None, label_col: str = "World Zone", title: str = "Clusters on Embedding Space", figsize: tuple = (8, 6),save_path: Path | None = None):
    """
    聚类结果叠加可视化
    Visualize clusters over embedding space.

    绘制聚类结果与降维结果的叠加散点图，
    并在每个簇中心标注主标签（Dominant Label）及纯度（Dominant %）。
    Overlays clustering results on embedding plots, labeling each cluster
    with its dominant class and purity percentage.

    :param embedding_df: pd.DataFrame
        降维结果。Embedding coordinates (must include “Dim1”, “Dim2”).

    :param labels: pd.Series
        聚类标签。Cluster labels.

    :param meta: pd.DataFrame | None
        样本元数据。Sample metadata (optional).

    :param label_col: str, default="World Zone"
        真实标签列名。Name of ground-truth label column.

    :param title: str
        图标题。Plot title.

    :param save_path: Path | None
        若指定路径则保存图片，否则直接显示。
        Save the plot if a path is provided; otherwise display interactively.

    :param figsize: tuple
        图像大小。Figure size.

    说明 / Notes:
        - 点颜色代表聚类簇。
          Point colors indicate cluster IDs.
        - 若提供 meta，可计算主标签及其占比（纯度）。
          If metadata is provided, computes dominant label and purity per cluster.
        - 输出出版级聚类分布可视化结果。
          Produces publication-ready cluster visualization.
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

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[OK] Cluster plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_silhouette_trend(scores: list[tuple[int, float]], save_path: Path | None = None) -> None:
    """
    绘制轮廓系数随聚类数变化的趋势图
    Plot silhouette score trend by number of clusters.

    :param scores: list of (k, silhouette_score)
        每个聚类数对应的得分。
        List of (k, silhouette_score) pairs.

    :param save_path: Path | None
        若指定路径则保存图片，否则直接显示。
        Save the plot if a path is provided; otherwise display interactively.
    """
    ks, vals = zip(*scores)
    plt.figure(figsize=(7, 4))
    plt.plot(ks, vals, marker="o", lw=2, color="#4c72b0")
    plt.title("Silhouette Score by Cluster Number")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[OK] Silhouette trend plot saved: {save_path}")
    else:
        plt.show()

    plt.close()