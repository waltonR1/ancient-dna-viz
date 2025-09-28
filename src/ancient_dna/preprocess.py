import numpy as np
import pandas as pd
from typing import Tuple


def align_by_id(ids: pd.Series, X: pd.DataFrame, meta: pd.DataFrame, id_col: str = "Genetic ID") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按样本 ID 对齐基因型矩阵与注释表，仅保留两表共有的样本，并保证行顺序一致。

    :param ids: 样本 ID 序列（与 X 的行一一对应）。
    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param meta: 注释表 (pd.DataFrame)，包含样本 ID 以及标签信息（如 Y haplogroup）。
    :param id_col: 注释表中的样本 ID 列名（默认 "Genetic ID"）。
    :return: (X_aligned, meta_aligned)
             - X_aligned: 仅保留共有样本后的基因型矩阵（重新从 0 开始索引）。
             - meta_aligned: 与 X_aligned 行顺序一致的注释表。
    注意:
        - 若注释表中缺失某些样本，将被自动丢弃；仅保留交集。
    """
    common = ids[ids.isin(set(meta[id_col]))]
    mask = ids.isin(set(common))
    X_aligned = X.loc[mask].reset_index(drop=True)
    meta_aligned = meta.set_index(id_col).loc[common].reset_index()
    return X_aligned, meta_aligned


def compute_missing_rates(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    计算缺失率（样本维度 & SNP 维度）。
        - 0 = 参考等位基因
        - 1 = 变异等位基因
        - 3 = 缺失

    :param X: 基因型矩阵 (pd.DataFrame)。
    :return: (sample_missing, snp_missing)
             - sample_missing: 每个样本（行）的缺失率 (0~1)。
             - snp_missing: 每个 SNP（列）的缺失率 (0~1)。
    """
    Z = X.replace(3, np.nan)
    sample_missing = Z.isna().mean(axis=1)
    snp_missing = Z.isna().mean(axis=0)
    return sample_missing, snp_missing


def filter_by_missing(
    X: pd.DataFrame,
    sample_missing: pd.Series,
    snp_missing: pd.Series,
    max_sample_missing: float = 0.8,
    max_snp_missing: float = 0.8
) -> pd.DataFrame:
    """
    按缺失率阈值过滤样本与 SNP（先跑通用的 baseline 阈值，后续可调严）。

    :param X: 基因型矩阵。
    :param sample_missing: 每个样本的缺失率 (pd.Series)。
    :param snp_missing: 每个 SNP 的缺失率 (pd.Series)。
    :param max_sample_missing: 样本级最大缺失率阈值（默认 0.8，保守）。
    :param max_snp_missing: SNP 级最大缺失率阈值（默认 0.8，保守）。
    :return: 过滤后的矩阵 (pd.DataFrame)，索引从 0 重新开始。
    """
    keep_rows = sample_missing <= max_sample_missing
    keep_cols = snp_missing[snp_missing <= max_snp_missing].index
    return X.loc[keep_rows, keep_cols].reset_index(drop=True)


def _fill_col_mode(col: pd.Series, fallback: float = 1) -> pd.Series:
    """
    列众数填补的内部工具函数。

    :param col: 一列 SNP 数据（可能包含 NaN）。
    :param fallback: 当整列均为空/无众数时的回退值（默认填 1）。
    :return: 填补后的该列。
    说明:
        - 众数 ties 时使用 value_counts 的默认顺序（出现次数最多的第一个）。
    """
    vc = col.dropna().value_counts()
    return col.fillna(vc.index[0] if len(vc) else fallback)


def impute_missing(X: pd.DataFrame, method: str = "mode") -> pd.DataFrame:
    """
    缺失值填补（提供列众数填补；后续可扩展 KNN/矩阵分解/自编码器等）。

    :param X: 基因型矩阵 (pd.DataFrame)。（编码规则: 0 = 参考等位基因, 1 = 变异等位基因, 3 = 缺失）
    :param method: 填补方法（当前实现 'mode'；其它方法请在后续版本扩展）。
    :return: 填补后的矩阵 (pd.DataFrame)。
    """
    Z = X.replace(3, np.nan)
    if method == "mode":
        return Z.apply(_fill_col_mode, axis=0)
    raise ValueError("Only 'mode' is implemented in Week1. 请在后续版本扩展 'knn' 等方法。")
