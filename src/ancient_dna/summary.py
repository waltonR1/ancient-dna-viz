"""
summary.py
-----------------
报告生成与导出模块
Report generation and export module.

用于生成与保存数据处理过程中的各类分析报告，
包括缺失率报告、降维结果报告以及运行时间报告。
Provides utilities to build and export analytical reports such as
missing rate summaries, embedding statistics, and runtime performance logs.

功能 / Functions:
    - build_missing_report(): 生成样本与 SNP 缺失率统计报告；
      Build a summary report for sample- and SNP-level missingness.
    - build_embedding_report(): 生成降维嵌入的数值分布报告；
      Build a statistical report for embedding dimensions.
    - save_report(): 通用报告保存函数（CSV 导出）；
      Save any report DataFrame as a UTF-8 CSV file.
    - save_runtime_report(): 保存运行时间报告（runtime_summary.csv）。
      Save algorithm runtime summary as a CSV file.

说明 / Description:
    本模块是分析管线的“报告层”（Reporting Layer），
    用于整合、统计与导出数据质量和模型性能结果。
    可与 preprocess、embedding、clustering 模块协同使用，
    为整个数据处理工作流提供可追踪、可审计的输出。
    This module serves as the "Reporting Layer" of the pipeline,
    consolidating, summarizing, and exporting quality and performance metrics.
    It works in conjunction with preprocess, embedding, and clustering modules,
    providing traceable and auditable outputs across the data workflow.
"""

import pandas as pd
from pathlib import Path
from ancient_dna import save_csv


def build_missing_report(sample_missing: pd.Series, snp_missing: pd.Series) -> pd.DataFrame:
    """
    生成缺失率汇总报告
    Build a summary report of missing rates.

    根据样本维度与 SNP 维度的缺失率计算描述性统计，
    输出包含样本数量、SNP 数量、均值、中位数及最大值的单行报告表。
    Computes descriptive statistics (mean, median, max) of missing rates
    at both the sample and SNP levels, returning a one-row summary DataFrame.

    :param sample_missing: pd.Series
        每个样本的缺失率。
        Missing rate per sample.

    :param snp_missing: pd.Series
        每个 SNP 的缺失率。
        Missing rate per SNP.

    :return: pd.DataFrame
        单行汇总表，包含样本与 SNP 层面的缺失率统计。
        One-row DataFrame summarizing missing rate statistics.

    说明 / Notes:
        - 汇总样本级与位点级缺失率指标；
          Summarizes missingness at both sample and site levels.
        - 计算均值、中位数、最大值，用于质量评估。
          Includes mean, median, and max values for quick quality assessment.
        - 可作为数据清洗阶段的监控指标。
          Useful for monitoring data quality before downstream processing.
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
    Build a statistical report for embedding results.

    计算降维结果中各维度的描述性统计（均值、标准差、最小值、最大值），
    可用于分析降维后特征分布与检测维度坍缩。
    Computes descriptive statistics (mean, std, min, max) for each embedding dimension,
    useful for evaluating numeric range and detecting dimensional collapse.

    :param embedding: pd.DataFrame
        降维后的嵌入结果，列名通常为 ["Dim1", "Dim2", ...]。
        The embedding matrix (columns typically "Dim1", "Dim2", ...).

    :return: pd.DataFrame
        每个维度的统计报告。
        Statistical summary per dimension.

    说明 / Notes:
        - 报告包括每个维度的均值、标准差、最小值与最大值。
          Reports mean, standard deviation, min, and max per dimension.
        - 可用于评估降维数值范围与稳定性。
          Helps assess the numeric range and stability of embedding.
        - 若某维度方差极低，可能暗示降维坍缩或特征冗余。
          Low variance in a dimension may indicate collapse or redundancy.
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

    将任意报告表（如缺失率报告或降维报告）保存为 CSV，
    自动创建目录并采用 UTF-8 编码。
    Saves any report table (e.g., missingness or embedding reports) as a UTF-8 CSV file,
    automatically creating parent directories if needed.

    :param df: pd.DataFrame
        报告表格。
        Report DataFrame.

    :param path: str | Path
        输出文件路径。
        Output file path.

    说明 / Notes:
        - 自动创建输出目录；
          Automatically creates parent directories if they do not exist.
        - 使用 UTF-8 编码导出；
          Exports CSV with UTF-8 encoding.
        - 输出包含列名。
          Column headers are always included in the output.
    """
    save_csv(df, path, verbose=False)
    print(f"[OK] The report is saved: {path}")


def save_runtime_report(records: list[dict], path: str | Path) -> None:
    """
    保存降维运行时间统计报告
    Save runtime summary report for embedding or imputation steps.

    将各算法运行时间记录导出为 runtime_summary.csv，
    用于性能比较或实验日志分析。
    Exports runtime logs of algorithms to a CSV file for benchmarking or performance tracking.

    :param records: list[dict]
        包含每个算法运行时间的字典列表，
        格式示例：[{"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52}]。
        List of runtime record dictionaries,
        e.g., [{"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52}].

    :param path: str | Path
        输出文件路径。
        Output file path.

    :return: None
        无返回值，直接写入文件。
        None (writes file directly).

    说明 / Notes:
        - 若 records 为空，将输出警告并跳过保存。
          If no records are provided, the function prints a warning and returns.
        - 自动创建输出目录并保存为 CSV 文件。
          Automatically saves the summary as a CSV file in UTF-8 encoding.
        - 可用于生成性能比较报告或实验结果摘要。
          Useful for building performance comparison summaries across methods.
    """
    print("[INFO] Collect the runtime summary:")
    if not records:
        print("[WARN] No runtime records to save.")
        return

    df = pd.DataFrame(records)
    save_csv(df, path, verbose=False)

    print(f"[OK] Runtime summary report saved: {path.resolve()} ({len(df)} rows)")