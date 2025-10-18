import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines
from pathlib import Path
from matplotlib.colors import ListedColormap
import numpy as np

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
        others_color : tuple = (0.7, 0.7, 0.7, 0.5)
) -> None:
    """
    绘制降维结果（支持 2D），点与图例颜色严格一致；
    超出 legend_max 的类别在图中与 legend 中均以灰色表示。

    :param df: 投影后的 DataFrame，至少包含 ["Dim1", "Dim2"]。
    :param labels: 分类标签 (pd.Series)，用于着色。可选。
    :param title: 图标题。
    :param save_path: 保存路径（如 "plot.png"）。若为 None，则直接显示。
    :param figsize: 图像大小，默认 (10, 7)。
    :param legend_pos: 图例位置，可选 {"right", "bottom", "top", "inside"}。
    :param cmap: 颜色映射表（默认 "tab20"）。
    :param legend_max: 图例显示的最大类别数（超过则合并为灰色 others）。
    :param legend_sort: 是否按样本数量排序（默认 True）。
    :param others_color:超出legend限制的样本的颜色

    说明：
        - 图中点与 legend 颜色保持一致；
        - 超出 legend_max 的类别在图中以灰色显示；
        - 使用固定离散色表 (tab20)；
        - 图例默认在右侧；
        - 支持保存或直接显示。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # ====== Step 1. 分类与排序 ======
        categories = pd.Categorical(labels)
        n_classes = len(categories.categories)

        if legend_sort:
            counts = pd.Series(categories).value_counts()
            ordered_categories = counts.index.tolist()  # 按样本数降序
            categories = pd.Categorical(labels, categories=ordered_categories, ordered=True)
        else:
            ordered_categories = list(categories.categories)

        # ====== Step 2. 使用 tab20 颜色 ======
        base_cmap = get_cmap(cmap)
        tab20_colors = [base_cmap(i / 19) for i in range(20)]  # tab20 自带 20 种颜色

        # ====== Step 3. 区分主要类别与 others ======
        if n_classes > legend_max:
            main_colors = tab20_colors[:legend_max]
            mask_main = categories.codes < legend_max
            # mask_others = ~mask_main

            # 主类散点
            ax.scatter(
                df.loc[mask_main, "Dim1"],
                df.loc[mask_main, "Dim2"],
                c=categories.codes[mask_main],
                cmap=ListedColormap(main_colors),
                s=20, alpha=0.5
            )

            # 其他类灰色(此处注释掉使得不绘制灰色的点)
            # ax.scatter(
            #     df.loc[mask_others, "Dim1"],
            #     df.loc[mask_others, "Dim2"],
            #     color=others_color,
            #     s=20, alpha=0.3
            # )

            display_classes = list(categories.categories[:legend_max]) + ["... (others)"]
            display_colors = main_colors + [others_color]

        else:
            main_colors = tab20_colors[:n_classes]
            ax.scatter(
                df["Dim1"], df["Dim2"],
                c=categories.codes,
                cmap=ListedColormap(main_colors),
                s=20, alpha=0.5
            )
            display_classes = categories.categories
            display_colors = main_colors

        # ====== Step 4. Legend 绘制 ======
        legend_title = labels.name if labels.name else "Category"
        handles = [
            mlines.Line2D([], [], color=display_colors[i], marker='o',
                          linestyle='None', markersize=6)
            for i in range(len(display_classes))
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

        ax.legend(
            handles,
            display_classes,
            title=legend_title,
            loc=loc,
            bbox_to_anchor=bbox,
            frameon=False,
            fontsize=9,
            title_fontsize=10
        )

    else:
        # ====== 无标签情况 ======
        ax.scatter(df["Dim1"], df["Dim2"], s=30, alpha=0.7, color="gray")
        rect = [0, 0, 1, 1]

    # ====== Step 5. 坐标轴与标题 ======
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_title(title)

    plt.tight_layout(rect = rect)

    # ====== Step 6. 保存或显示 ======
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] 图像已保存到 {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_missing_values(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    missing_value: int = 3,
    figsize: tuple = (20, 10),
    cmap_present: str = "#d95f02",
    cmap_missing: str = "#ffffff",
    show_ratio: bool = True,
) -> None:
    """
    绘制缺失数据可视化图（白色 = 缺失）。

    :param df: 基因样本数据。
    :param save_path: 保存路径（如 "plot.png"）。若为 None，则直接显示。
    :param missing_value: 缺失值标记（默认 3）。
    :param figsize: 图像大小 (宽, 高)，默认 (20, 10)。
    :param cmap_present: 非缺失值颜色，默认橙色 (#d95f02)。
    :param cmap_missing: 缺失值颜色，默认白色 (#ffffff)。
    :param show_ratio: 是否同时显示缺失比例条形图，默认 True。
    """
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
        print(f"[OK] 图像已保存到 {save_path}")
    else:
        plt.show()