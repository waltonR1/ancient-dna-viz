"""
summary.py
==========

报告生成与导出模块（Reporting Layer）
Report generation and export module.

本模块用于生成并导出数据处理流程中的各类分析报告，
包括缺失率统计、降维嵌入数值分布以及算法运行时间摘要。

This module provides utilities to generate and export analytical
reports produced during the data processing pipeline, including
missing-rate summaries, embedding statistics, and runtime logs.

Functions
---------
build_missing_report
    生成样本级与 SNP 级缺失率的统计汇总报告。
    Build a summary report of sample- and SNP-level missingness.

build_embedding_report
    生成降维嵌入结果的数值分布统计报告。
    Build a statistical summary for embedding dimensions.

save_report
    通用报告保存函数（CSV 导出）。
    Save a report DataFrame as a CSV file.

save_runtime_report
    保存算法运行时间的汇总报告。
    Save runtime summary records as a CSV file.

Description
-----------
本模块是分析管线的“报告层（Reporting Layer）”，
负责对 preprocess、embedding、clustering 等模块
产生的中间结果进行汇总、统计与持久化。

This module serves as the reporting layer of the pipeline,
consolidating, summarizing, and exporting results produced
by preprocess, embedding, and clustering stages, and providing
traceable, auditable outputs for downstream analysis.
"""

import pandas as pd
from pathlib import Path
from ancient_dna import save_csv


def build_missing_report(
        sample_missing: pd.Series,
        snp_missing: pd.Series
) -> pd.DataFrame:
    """
    生成缺失率汇总报告

    Build a summary report of missing rates.

    本函数根据样本维度与 SNP 维度的缺失率，
    计算描述性统计指标，并返回一个单行汇总表，
    用于概览数据的缺失情况。

    This function computes descriptive statistics of missing
    rates at both the sample and SNP levels and returns a
    single-row summary DataFrame.

    Parameters
    ----------
    sample_missing : pandas.Series
        每个样本的缺失率。

        Missing rate per sample.

    snp_missing : pandas.Series
        每个 SNP 的缺失率。

        Missing rate per SNP.

    Returns
    -------
    pandas.DataFrame
        单行汇总表，包含样本与 SNP 层面的
        缺失率统计信息。

        One-row DataFrame summarizing missing-rate statistics
        at both the sample and SNP levels.

    Notes
    -----
    - 统计指标基于 ``Series.describe()`` 计算，
      包括均值（mean）、中位数（50%）和最大值（max）。

      Statistics are derived from ``Series.describe()``,
      including mean, median (50%), and max values.

    - 返回结果为单行 DataFrame，
      适合用于日志记录或后续汇总。

      The returned DataFrame contains a single row and is
      suitable for logging or downstream aggregation.
    """

    sm = sample_missing.describe()
    cm = snp_missing.describe()

    report = pd.DataFrame({
        "sample_count": [len(sample_missing)],
        "snp_count": [len(snp_missing)],
        "sample_missing_mean": [sm["mean"]],
        "sample_missing_median": [sm["50%"]],
        "sample_missing_max": [sm["max"]],
        "snp_missing_mean": [cm["mean"]],
        "snp_missing_median": [cm["50%"]],
        "snp_missing_max": [cm["max"]],
    })

    print("[INFO] Build missing report:")

    return report


def build_embedding_report(embedding: pd.DataFrame) -> pd.DataFrame:
    """
    生成降维嵌入结果的统计报告

    Build a statistical summary report for embedding results.

    本函数对降维后的 embedding 矩阵按维度计算
    描述性统计指标，并返回每个维度一行的汇总表。

    This function computes descriptive statistics for each
    dimension of an embedding matrix and returns a per-dimension
    summary table.

    Parameters
    ----------
    embedding : pandas.DataFrame
        降维后的嵌入矩阵，列表示嵌入维度，
        列名通常为 ``Dim1``, ``Dim2`` 等。

        Embedding matrix where columns correspond to
        embedding dimensions.

    Returns
    -------
    pandas.DataFrame
        每个嵌入维度一行的统计报告，
        包含均值、标准差、最小值和最大值，
        并将维度名称作为普通列返回。

        Statistical summary per embedding dimension,
        with one row per dimension and the dimension
        name returned as a column.

    Notes
    -----
    - 统计指标基于 ``DataFrame.describe()`` 计算，
      并保留均值（Mean）、标准差（StdDev）、
      最小值（Min）和最大值（Max）。

      Statistics are derived from ``DataFrame.describe()``
      and include mean, standard deviation, minimum, and maximum.

    - 返回结果通过 ``reset_index()`` 展平索引，
      便于后续导出或拼接。

      The index is reset before returning to facilitate
      export and downstream aggregation.

    - 该报告可作为分析 embedding 数值分布的
      辅助信息，但不包含任何自动判定逻辑。

      The report provides descriptive statistics only
      and does not perform automatic validity checks.
    """

    print("[INFO] Build embedding report:")
    stats = embedding.describe().T[["mean", "std", "min", "max"]]
    stats = stats.rename(columns={
        "mean": "Mean",
        "std": "StdDev",
        "min": "Min",
        "max": "Max"
    })
    stats.index.name = "Dimension"
    print("[OK] Embedding report built.")
    return stats.reset_index()


def save_report(df: pd.DataFrame, path: str | Path) -> None:
    """
    保存报告表格为 CSV 文件

    Save a report DataFrame as a CSV file.

    本函数是对 ``save_csv`` 的轻量封装，
    用于将分析报告表格导出为 CSV 文件，
    并统一日志输出格式。

    This function is a thin wrapper around ``save_csv`` for
    exporting analysis report tables as CSV files with
    consistent logging.

    Parameters
    ----------
    df : pandas.DataFrame
        需要保存的报告表格。

        Report DataFrame to be saved.

    path : str or pathlib.Path
        输出 CSV 文件路径。

        Output path of the CSV file.

    Returns
    -------
    None

    Notes
    -----
    - 文件写入行为（如编码方式、是否创建目录）
      由 ``save_csv`` 的具体实现决定。

      File writing behavior (e.g. encoding, directory creation)
      is delegated to the implementation of ``save_csv``.

    - 本函数仅负责调用 ``save_csv`` 并输出状态日志，
      不对数据内容做任何修改。

      This function only handles delegation and logging
      and does not modify the data.
    """

    save_csv(df, path, verbose=False)
    print(f"[OK] The report is saved: {path}")


def save_runtime_report(records: list[dict], path: str | Path) -> None:
    """
    保存算法运行时间汇总报告

    Save a runtime summary report as a CSV file.

    本函数将一组运行时间记录字典整理为表格，
    并通过 ``save_csv`` 导出为 CSV 文件，
    用于保存实验或流程的运行时间日志。

    This function converts a list of runtime record dictionaries
    into a tabular format and exports it as a CSV file via
    ``save_csv`` for logging and analysis.

    Parameters
    ----------
    records : list of dict
        运行时间记录列表，每个元素为一个字典。
        本函数不对字典结构做强制约束。

        List of runtime record dictionaries. The structure of
        each dictionary is not enforced by this function.

    path : str or pathlib.Path
        输出 CSV 文件路径。

        Output path of the CSV file.

    Returns
    -------
    None

    Notes
    -----
    - 若 ``records`` 为空列表，
      本函数将打印警告信息并直接返回，
      不会创建输出文件。

      If ``records`` is empty, the function prints a warning
      and returns without writing a file.

    - 文件写入行为（如编码方式、目录创建）
      由 ``save_csv`` 的具体实现决定。

      File writing behavior (e.g. encoding, directory creation)
      is delegated to the implementation of ``save_csv``.

    - 本函数仅负责汇总与保存记录，
      不对运行时间数据进行校验或解释。

      This function only handles aggregation and saving
      and does not validate or interpret runtime values.
    """

    print("[INFO] Collect the runtime summary:")
    if not records:
        print("[WARN] No runtime records to save.")
        return

    df = pd.DataFrame(records)
    save_csv(df, path, verbose=False)

    print(f"[OK] Runtime summary report saved: {path.resolve()} ({len(df)} rows)")