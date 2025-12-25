"""
visualize.py
============

可视化模块（Visualization Layer）
Visualization and plotting utilities.

本模块负责对降维 embedding、缺失值模式、聚类结果及标签信息进行可视化展示，
包括：静态 Matplotlib 绘图、交互式 Plotly 绘图、类别颜色映射与 World Zone 前缀分组等。

This module provides visualization utilities for embedding projections,
missing-value patterns, clustering overlays, and categorical labels,
including static Matplotlib plots, interactive Plotly plots, categorical
color mapping, and World Zone prefix grouping.

Functions
---------
_is_world_zone_labels
    判断类别标签是否符合 World Zone 前缀语义。
    Determine whether labels follow a World Zone prefix convention.

_group_by_world_zone
    根据 World Zone 前缀对类别进行分组。
    Group category labels by World Zone prefix.

_categorical_color_mapping
    为离散类别生成颜色映射，并区分 main / others 类别。
    Generate color mapping for categorical labels with main/others handling.

plot_embedding
    使用 Matplotlib 绘制 2D / 3D embedding（可选标签着色与图例）。
    Plot 2D/3D embeddings using Matplotlib with optional categorical coloring.

plot_embedding_interactive
    使用 Plotly 绘制交互式 2D / 3D embedding（每类单独 trace）。
    Plot interactive 2D/3D embeddings using Plotly (one trace per class).

plot_missing_values
    绘制缺失值模式：小矩阵逐像素图；大矩阵缺失率分布直方图。
    Visualize missingness: pixel pattern for small matrices; histograms for large matrices.

plot_cluster_on_embedding
    在 embedding 空间叠加聚类结果，并可标注每簇主标签与纯度。
    Overlay cluster assignments on embedding and optionally annotate dominant labels/purity.

plot_silhouette_trend
    绘制 silhouette score 随聚类数 k 的变化趋势。
    Plot silhouette score trend as a function of number of clusters.

Description
-----------
本模块属于分析流程的“可视化层（Visualization Layer）”，
用于将 embedding、聚类与元数据标签直观呈现，以支持探索分析与结果解释。

This module serves as the visualization layer of the analysis pipeline,
rendering embeddings, clusters, and metadata labels for exploratory analysis
and interpretation.
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
    "Africa": get_cmap("Purples"),
    "America": get_cmap("Reds"),
    "Oceania": get_cmap("vanimo"),
    "Middle East": get_cmap("Greys"),
}


_WORLD_ZONE_ORDER = [
    "Europe",
    "Asia",
    "Middle East",
    "Africa",
    "America",
    "Oceania",
]


def _is_world_zone_labels(classes: list[str]) -> bool:
    """
    判断类别标签是否符合 World Zone 前缀语义
    Determine whether labels follow a World Zone prefix convention.

    本函数检查给定的类别名称列表中，
    是否有至少 60% 的类别名称
    以已定义的 World Zone 名称作为字符串前缀
    （如 "Europe", "Asia", "Africa" 等）。

    This function checks whether at least 60% of the given
    class names start with a predefined World Zone prefix
    (e.g. "Europe", "Asia", "Africa").

    Parameters
    ----------
    classes : list of str
        类别名称列表。

        List of class names.

    Returns
    -------
    bool
        若满足 World Zone 前缀比例阈值（≥ 60%）则返回 True，
        否则返回 False。

        Returns True if the proportion of class names
        matching World Zone prefixes meets or exceeds
        the threshold; otherwise returns False.

    Notes
    -----
    - 判断基于字符串前缀匹配（``str.startswith``），
      不解析或规范化类别名称。

      Matching is based on string prefix comparison only
      and does not perform normalization or parsing.

    - World Zone 前缀集合来源于 ``_WORLD_ZONE_CMAPS`` 的键。

      The set of World Zone prefixes is taken from the
      keys of ``_WORLD_ZONE_CMAPS``.

    - 当 ``classes`` 为空列表时，直接返回 False。

      Returns False immediately if the input list is empty.
    """

    if not classes:
        return False
    return sum(
        any(cls.startswith(cont) for cont in _WORLD_ZONE_CMAPS)
        for cls in classes
    ) / len(classes) >= 0.6


def _group_by_world_zone(classes: list[str]) -> dict[str, list[str]]:
    """
    将类别标签按 World Zone 前缀进行分组
    Group class labels by World Zone prefix.

    本函数根据预定义的 World Zone 顺序，
    使用字符串前缀匹配的方式，
    将类别名称分组到对应的 World Zone 下。
    未匹配任何前缀的类别将被归入 "Other" 组。

    This function groups class names into World Zones
    based on string prefix matching, following a predefined
    World Zone order. Class names that do not match any
    prefix are assigned to the "Other" group.

    Parameters
    ----------
    classes : list of str
        类别名称列表。

        List of class names.

    Returns
    -------
    dict[str, list[str]]
        以 World Zone 名称为键、类别名称列表为值的分组字典。
        未匹配的类别统一归入键 "Other"。

        Dictionary mapping World Zone names to lists of
        class names. Unmatched classes are grouped under
        the "Other" key.

    Notes
    -----
    - 匹配方式基于 ``str.startswith``，
      不进行任何字符串规范化或语义解析。

      Matching is based solely on ``str.startswith`` and
      does not perform normalization or semantic parsing.

    - World Zone 的匹配顺序由 ``_WORLD_ZONE_ORDER`` 决定，
      每个类别只会被分配到第一个匹配到的分组中。

      The matching order follows ``_WORLD_ZONE_ORDER``,
      and each class is assigned to the first matching group only.

    - 输出字典中的 "Other" 键在需要时自动创建，
      即使输入中未显式包含该类别。

      The "Other" key is created implicitly when needed.
    """

    groups = defaultdict(list)

    for cls in classes:
        matched = False
        for cont in _WORLD_ZONE_ORDER:
            if cls.startswith(cont):
                groups[cont].append(cls)
                matched = True
                break
        if not matched:
            groups["Other"].append(cls)

    return groups


def _categorical_color_mapping(
        labels: pd.Series,
        legend_max: int = 20,
        legend_sort: bool = True,
        cmap: str = "tab20",
        others_color: tuple = (0.8, 0.8, 0.8, 1.0),
        shade_range: tuple = (0.4, 0.85),
) -> tuple:
    """
    为离散类别生成颜色映射方案
    Generate a color mapping for categorical labels.

    本函数根据输入的类别标签生成用于可视化的颜色映射，
    支持普通离散类别映射与 World Zone 前缀语义映射两种模式，
    并对超出图例数量限制的类别统一归并为 "others"。

    This function generates a color mapping for categorical labels,
    supporting both standard categorical mapping and World Zone
    prefix-based mapping, with excess classes merged into "others".

    Parameters
    ----------
    labels : pandas.Series
        类别标签序列，用于生成颜色映射。

        Series of categorical labels.

    legend_max : int, default=20
        主图例中最多显示的类别数量。
        超出部分将被归为 "others"。

        Maximum number of classes shown in the main legend.

    legend_sort : bool, default=True
        是否按类别出现频次对类别进行排序。
        若为 False，则保持标签首次出现顺序。

        Whether to sort classes by frequency or preserve
        the order of first appearance.

    cmap : str, default="tab20"
        普通离散类别使用的 Matplotlib colormap 名称。

        Name of the Matplotlib colormap used for standard
        categorical coloring.

    others_color : tuple
        RGBA 颜色值，用于表示被归并的 "others" 类别。

        RGBA color used for merged "others" classes.

    shade_range : tuple
        World Zone 模式下，同一 Zone 内颜色渐变的取值范围。

        Range of colormap values used to generate shades
        within each World Zone.

    Returns
    -------
    tuple
        返回一个四元组 ``(class_to_color, main_classes, other_classes, main_colors)``：

        - class_to_color : dict[str, tuple]
            每个类别到 RGBA 颜色的映射。
        - main_classes : list[str]
            主图例中显示的类别列表。
        - other_classes : list[str]
            被归并为 "others" 的类别列表。
        - main_colors : list[tuple]
            与 ``main_classes`` 一一对应的颜色列表。

    Notes
    -----
    - World Zone 模式通过 ``_is_world_zone_labels`` 判定，
      并基于类别名称的字符串前缀进行分组。

      World Zone mode is detected via ``_is_world_zone_labels``
      and relies on string prefix matching.

    - 在 World Zone 模式下，
      类别分组顺序由 ``_WORLD_ZONE_ORDER`` 决定，
      每个 Zone 内部类别按字母顺序排列。

      In World Zone mode, grouping follows
      ``_WORLD_ZONE_ORDER`` and classes within each zone
      are sorted alphabetically.

    - 本函数不对标签进行规范化或校验，
      所有处理均基于字符串形式的原始标签。

      No normalization or validation is performed on labels;
      all operations use their string representations.
    """

    labels = labels.astype(str)

    # 排序
    ordered = (
        labels.value_counts().index.tolist()
        if legend_sort
        else labels.unique().tolist()
    )

    main_classes = ordered[:legend_max]
    other_classes = ordered[legend_max:]

    class_to_color = {}

    # World Zone 语义
    if _is_world_zone_labels(main_classes):
        groups = _group_by_world_zone(main_classes)

        ordered_main = []
        main_colors = []

        for cont in _WORLD_ZONE_ORDER + ["Other"]:
            if cont not in groups:
                continue

            classes = sorted(groups[cont])
            cmap_cont = _WORLD_ZONE_CMAPS.get(cont, get_cmap(cmap))
            shades = np.linspace(*shade_range, len(classes))

            for cls, shade in zip(classes, shades):
                color = cmap_cont(shade)
                class_to_color[cls] = color
                ordered_main.append(cls)
                main_colors.append(color)

        main_classes = ordered_main

    # 普通 categorical
    else:
        base_cmap = get_cmap(cmap)
        colors = [
            base_cmap(i / max(1, len(main_classes) - 1))
            for i in range(len(main_classes))
        ]
        class_to_color.update(dict(zip(main_classes, colors)))
        main_colors = colors

    # Others
    for cls in other_classes:
        class_to_color[cls] = others_color

    return class_to_color, main_classes, other_classes, main_colors


def plot_embedding(
        df: pd.DataFrame,
        labels: pd.Series | None = None,
        title: str = "Projection",
        save_path: str | Path | None = None,
        figsize: tuple = (10, 7),
        legend_pos: str = "right",
        cmap: str = "tab20",
        legend_max: int = 20,
        legend_sort: bool = True,
        others_color: tuple = (0.7, 0.7, 0.7, 0.5),
        draw_others: bool = False,
) -> None:
    """
    绘制降维 embedding 的二维或三维可视化结果
    Plot 2D or 3D embedding projections.

    本函数基于 Matplotlib 绘制 embedding 的散点图，
    自动根据输入列判断是二维还是三维投影。
    若提供分类标签，则按类别进行颜色映射并绘制图例。

    This function renders a scatter plot of an embedding using Matplotlib,
    automatically switching between 2D and 3D projections based on
    the presence of a ``Dim3`` column. Optional categorical labels
    can be provided for color mapping and legend display.

    Parameters
    ----------
    df : pandas.DataFrame
        降维后的 embedding 数据，
        必须包含 ``Dim1`` 和 ``Dim2``，
        三维情况下需额外包含 ``Dim3``。

        Embedding DataFrame containing ``Dim1`` and ``Dim2``,
        and optionally ``Dim3`` for 3D plots.

    labels : pandas.Series or None, optional
        用于着色的分类标签。
        若为 None，则绘制单色散点图。

        Optional categorical labels for color mapping.

    title : str, default="Projection"
        图标题。

        Plot title.

    save_path : str or pathlib.Path or None, optional
        若提供路径，则保存图像到该位置；
        否则直接以交互方式显示图像。

        If provided, the figure is saved to this path;
        otherwise, it is shown interactively.

    figsize : tuple, default=(10, 7)
        图像尺寸（单位：英寸）。

        Figure size in inches.

    legend_pos : str, default="right"
        图例位置，可选：
        ``"right"``, ``"bottom"``, ``"top"``, ``"inside"``。

        Legend position.

    cmap : str, default="tab20"
        普通离散类别使用的 Matplotlib colormap 名称。

        Colormap name used for categorical coloring.

    legend_max : int, default=20
        图例中最多显示的类别数量，
        超出部分归并为 "others"。

        Maximum number of classes displayed in the legend.

    legend_sort : bool, default=True
        是否按类别出现频次排序。

        Whether to sort legend classes by frequency.

    others_color : tuple
        RGBA 颜色值，用于表示被归并的 "others" 类别。

        RGBA color for merged "others" classes.

    draw_others : bool, default=False
        是否在图中绘制被归并为 "others" 的点。

        Whether to draw points belonging to merged "others" classes.

    Returns
    -------
    None

    Notes
    -----
    - 类别颜色映射由 ``_categorical_color_mapping`` 生成，
      本函数不直接参与颜色分配逻辑。

      Color assignment is delegated to ``_categorical_color_mapping``.

    - 当 ``labels`` 为 None 时，
      图中不显示图例，所有点使用统一颜色。

      If ``labels`` is None, no legend is shown and
      points are plotted in a single color.

    - 函数在绘制完成后始终关闭 figure，
      以避免 Matplotlib 资源泄漏。

      The figure is always closed after rendering to
      avoid resource leakage.
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
        # 无标签情况
        if is_3d:
            ax.scatter(df["Dim1"], df["Dim2"], df["Dim3"],
                       s=20, alpha=0.7, color="gray")
        else:
            ax.scatter(df["Dim1"], df["Dim2"],
                       s=20, alpha=0.7, color="gray")
        rect = [0, 0, 1, 1]

    # 坐标轴与标题
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

    # =保存或显示
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
        save_path: str | Path | None = None,
):
    """
    绘制交互式 embedding 可视化（Plotly）
    Plot an interactive embedding visualization using Plotly.

    本函数基于 Plotly 绘制 embedding 的二维或三维交互式散点图，
    每个类别对应一个独立的 trace，
    从而支持通过图例进行显示/隐藏控制。

    This function renders an interactive 2D or 3D scatter plot
    of an embedding using Plotly. Each category is drawn as a
    separate trace, enabling interactive legend toggling.

    Parameters
    ----------
    df : pandas.DataFrame
        降维后的 embedding 数据，
        必须包含 ``Dim1``、``Dim2``，
        三维情况下需包含 ``Dim3``。

        Embedding DataFrame containing ``Dim1`` and ``Dim2``,
        and optionally ``Dim3`` for 3D plots.

    labels : pandas.Series
        用于着色的分类标签。

        Categorical labels for color mapping.

    dim : int or None, optional
        指定 embedding 维度（2 或 3）。
        若为 None，则根据是否存在 ``Dim3`` 自动判断。

        Embedding dimensionality (2 or 3). If None, the
        dimensionality is inferred from the DataFrame.

    title : str, default="Projection"
        图标题。

        Plot title.

    legend_max : int, default=20
        图例中最多显示的类别数量，
        超出部分合并为 "others"。

        Maximum number of classes displayed in the legend.

    legend_sort : bool, default=False
        是否按类别出现频次排序（传递给颜色映射函数）。

        Whether to sort classes by frequency before color mapping.

    cmap : str, default="tab20"
        普通离散类别使用的 Matplotlib colormap 名称。

        Colormap name used as fallback for categorical coloring.

    others_color : tuple
        RGBA 颜色值，用于表示被归并的 "others" 类别。

        RGBA color for merged "others" classes.

    save_path : str or pathlib.Path or None, optional
        若提供路径，则将图像保存为 HTML 文件；
        否则直接以交互方式显示。

        If provided, the figure is saved as an HTML file;
        otherwise, it is shown interactively.

    Returns
    -------
    plotly.graph_objects.Figure
        生成的 Plotly Figure 对象。

        The generated Plotly Figure object.

    Notes
    -----
    - 类别颜色映射复用 ``_categorical_color_mapping``，
      并将 Matplotlib RGBA 颜色转换为 CSS ``rgba(...)`` 字符串。

      Color mapping is reused from ``_categorical_color_mapping``
      and converted from Matplotlib RGBA to CSS ``rgba(...)`` strings.

    - 每个类别对应一个独立的 Plotly trace，
      是交互式图例能够单独控制类别显示的基础。

      Each category is rendered as a separate Plotly trace,
      enabling interactive legend control.

    - 本函数不对 embedding 或标签进行校验或重排，
      假定其索引已对齐。

      The function assumes that embedding and labels are
      already aligned and performs no validation.
    """

    if dim is None:
        dim = 3 if "Dim3" in df.columns else 2

    labels = labels.astype(str)

    # Reuse categorical color mapping
    class_to_color, main_classes, other_classes, _ = _categorical_color_mapping(
        labels=labels,
        legend_max=legend_max,
        legend_sort=legend_sort,
        cmap=cmap,
        others_color=others_color,
    )

    fig = go.Figure()

    # One trace per class (key for interactive legend)
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

    # Others
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

    # Layout
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

    # Output
    if save_path:
        fig.write_html(str(save_path))
        print(f"[OK] Interactive figure saved → {save_path}")
    else:
        fig.show()
        print("[OK} Figure shown interactively.")

    return fig


def plot_missing_values(
        df: pd.DataFrame,
        save_path: str | Path | None = None,
        missing_value: int = 3,
        figsize: tuple = (20, 10),
        cmap_present: str = "#d95f02",
        cmap_missing: str = "#ffffff",
        show_ratio: bool = True,
        max_pixels: int = 5e7,
) -> None:
    """
    绘制缺失值分布与模式的可视化结果
    Visualize missing-value patterns and distributions.

    本函数根据矩阵规模自动选择可视化方式：
    对于较小矩阵，绘制逐像素的缺失模式图；
    对于较大矩阵，绘制缺失率分布的聚合直方图，
    以避免内存与渲染开销过大。

    This function automatically selects a visualization mode
    based on matrix size: pixel-wise missingness visualization
    for small matrices, and aggregated missing-rate histograms
    for large matrices.

    Parameters
    ----------
    df : pandas.DataFrame
        输入数据矩阵。

        Input data matrix.

    save_path : str or pathlib.Path or None, optional
        若提供路径，则保存图像到该位置；
        否则直接显示图像。

        If provided, the figure is saved to this path;
        otherwise, it is shown interactively.

    missing_value : int, default=3
        用于表示缺失值的编码。

        Value representing missing entries.

    figsize : tuple, default=(20, 10)
        图像尺寸（单位：英寸）。

        Figure size in inches.

    cmap_present : str
        非缺失值对应的颜色。

        Color used for present (non-missing) values.

    cmap_missing : str
        缺失值对应的颜色。

        Color used for missing values.

    show_ratio : bool, default=True
        在逐像素模式下，是否在下方绘制
        每列缺失率柱状图。

        Whether to display per-column missing-rate bars
        in pixel-wise mode.

    max_pixels : int
        矩阵元素数量阈值。
        当 ``df.shape[0] * df.shape[1]`` 超过该值时，
        自动切换为聚合可视化模式。

        Threshold on the total number of matrix elements
        for switching to aggregated visualization mode.

    Returns
    -------
    None

    Notes
    -----
    - 聚合模式下，绘制样本级缺失率直方图，
      以及 SNP 级缺失率直方图（最多随机采样 5000 列）。

      In aggregated mode, histograms are plotted for
      sample-level missing rates and SNP-level missing rates
      (with up to 5000 columns sampled).

    - 逐像素模式使用 ``imshow`` 绘制缺失掩码，
      并可选附加每列缺失率柱状图。

      Pixel-wise mode visualizes the missing-value mask
      using ``imshow``, with an optional per-column
      missing-rate bar chart.

    - 本函数仅用于可视化，不修改输入数据。

      This function performs visualization only and does
      not modify the input data.
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


def plot_cluster_on_embedding(
        embedding_df: pd.DataFrame,
        labels: pd.Series,
        meta: pd.DataFrame | None = None,
        label_col: str = "World Zone",
        title: str = "Clusters on Embedding Space",
        figsize: tuple = (8, 6),
        save_path: Path | None = None,
):
    """
    在二维 embedding 空间中可视化聚类结果
    Visualize cluster assignments on a 2D embedding space.

    本函数仅支持二维 embedding，
    在 ``Dim1``–``Dim2`` 平面中绘制聚类结果的散点图，
    并使用聚类标签作为颜色编码。
    可选地，在每个簇的中心位置标注其主标签及对应占比。

    This function supports **2D embeddings only** and visualizes
    clustering results on the ``Dim1``–``Dim2`` plane. Cluster
    labels are used for color encoding, and dominant labels with
    proportions can be optionally annotated at cluster centers.

    Parameters
    ----------
    embedding_df : pandas.DataFrame
        二维 embedding 数据，
        必须且仅需包含 ``Dim1`` 和 ``Dim2`` 两列。

        2D embedding DataFrame containing ``Dim1`` and ``Dim2``.

    labels : pandas.Series
        聚类标签，用于颜色编码。

        Cluster assignment labels used for coloring.

    meta : pandas.DataFrame or None, optional
        样本元数据表。
        若提供且包含 ``label_col``，
        则用于计算每个簇的主标签及其占比。

        Optional sample metadata used to compute dominant labels
        and proportions per cluster.

    label_col : str, default="World Zone"
        ``meta`` 中表示真实标签的列名。

        Name of the column in ``meta`` containing ground-truth labels.

    title : str, default="Clusters on Embedding Space"
        图标题。

        Plot title.

    figsize : tuple, default=(8, 6)
        图像尺寸（单位：英寸）。

        Figure size in inches.

    save_path : pathlib.Path or None, optional
        若提供路径，则保存图像；
        否则直接显示图像。

        If provided, the plot is saved to this path;
        otherwise, it is shown interactively.

    Returns
    -------
    None

    Notes
    -----
    - 本函数 **仅使用 ``Dim1`` 和 ``Dim2``**，
      不支持三维 embedding，可视化前需自行降至二维。

      This function uses **only ``Dim1`` and ``Dim2``** and does
      not support 3D embeddings.

    - 点的颜色直接对应聚类标签 ``labels``，
      并使用固定的 ``tab20`` colormap。

      Point colors directly correspond to cluster labels
      and use the fixed ``tab20`` colormap.

    - 当 ``meta`` 或 ``label_col`` 不可用时，
      不进行主标签与占比的计算与标注。

      If ``meta`` or ``label_col`` is unavailable,
      no dominant-label annotation is performed.

    - 本函数假定 ``embedding_df``、``labels`` 与 ``meta``（若提供）
      在行顺序上已经对齐，不进行一致性校验。

      The function assumes row alignment between inputs
      and performs no validation.
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
        meta["cluster"] = labels.values

        # 计算簇中心坐标
        group_centers = (
            embedding_df.assign(cluster=labels)
            .groupby("cluster")[["Dim1", "Dim2"]]
            .mean()
        )

        # 计算每簇主标签及纯度
        cluster_stats = (
            meta.groupby(["cluster", label_col])
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


def plot_silhouette_trend(
        scores: list[tuple[int, float]],
        save_path: Path | None = None
) -> None:
    """
    绘制轮廓系数随聚类数变化的趋势图
    Plot silhouette score trend by number of clusters.

    本函数根据给定的 ``(k, silhouette_score)`` 序列，
    绘制轮廓系数随聚类数变化的折线趋势图，
    仅用于可视化展示，不执行任何分析或排序逻辑。

    This function visualizes the trend of silhouette scores
    as a function of the number of clusters using a line plot.
    It performs visualization only and does not apply any
    analysis or reordering.

    Parameters
    ----------
    scores : list of tuple (int, float)
        聚类数与对应轮廓系数的列表，
        每个元素为 ``(k, silhouette_score)``。

        List of ``(k, silhouette_score)`` pairs.

    save_path : pathlib.Path or None, optional
        若提供路径，则保存图像；
        否则直接显示图像。

        If provided, the plot is saved to this path;
        otherwise, it is shown interactively.

    Returns
    -------
    None

    Notes
    -----
    - 输入顺序将被原样保留，
      本函数不会对 ``scores`` 进行排序或校验。

      The input order is preserved; no sorting or validation
      is performed.

    - 本函数仅负责绘制趋势图，
      不包含任何“最佳聚类数”的判定逻辑。

      This function only visualizes the trend and does not
      determine an optimal number of clusters.
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