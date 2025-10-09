import pandas as pd
from pathlib import Path


def build_missing_report(sample_missing: pd.Series, snp_missing: pd.Series) -> pd.DataFrame:
    """
    生成缺失率汇总报告。

    :param sample_missing: 每个样本的缺失率 (pd.Series)。
    :param snp_missing: 每个 SNP 的缺失率 (pd.Series)。
    :return: 单行 DataFrame，包含描述性统计结果。
    说明:
        - 汇总样本级与位点级的缺失率指标；
        - 包含均值、中位数、最大值等；
        - 可与 save_report() 搭配使用导出 CSV。
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

    return report


def build_embedding_report(embedding: pd.DataFrame) -> pd.DataFrame:
    """
    生成降维嵌入结果的统计报告。

    :param embedding: 降维后的嵌入结果 (pd.DataFrame)，列名通常为 ["Dim1", "Dim2", ...]。
    :return: 嵌入维度的统计报告 (pd.DataFrame)。
    说明:
        - 计算每个维度的均值、标准差、最小值、最大值；
        - 可用于评估降维结果的数值范围与分布；
        - 若某维方差过小，可能存在坍缩问题。
    """
    stats = embedding.describe().T[["mean", "std", "min", "max"]]
    stats = stats.rename(columns={
        "mean": "Mean",
        "std": "StdDev",
        "min": "Min",
        "max": "Max"
    })
    stats.index.name = "Dimension"
    return stats.reset_index()


def save_report(df: pd.DataFrame, path: str | Path) -> None:
    """
    保存报告表格为 CSV 文件。

    :param df: 报告 DataFrame。
    :param path: 保存路径。
    说明:
        - 自动创建上级目录；
        - 使用 UTF-8 编码；
        - 输出包含列名。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[OK] 报告已保存: {path}")


def combine_reports(reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    合并多份报告（例如不同阶段或不同指标）。

    :param reports: 报告字典，键为报告名称，值为 DataFrame。
    :return: 合并后的报告总表。
    说明:
        - 自动为每份报告添加前缀；
        - 用于生成多模块汇总表。
    """
    combined = []
    for name, df in reports.items():
        renamed = df.add_prefix(f"{name}_")
        combined.append(renamed)
    return pd.concat(combined, axis=1)


def save_runtime_report(records: list[dict], path: str | Path) -> None:
    """
    保存降维运行时间统计报告（runtime_summary.csv）

    :param records: 包含每个算法运行时间的字典列表。
                    格式示例：[{"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52}]
    :param path: 输出文件路径。
    :return: None
    """
    if not records:
        print("[WARN] No runtime records to save.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(path, sep=",", index=False, encoding="utf-8")

    print(f"[OK] Runtime summary report saved: {path.resolve()} ({len(df)} rows)")