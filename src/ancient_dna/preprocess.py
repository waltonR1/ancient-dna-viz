import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.impute import KNNImputer

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
    # 将编码 3 替换为 NaN 以便统计缺失率
    Z = X.replace(3, np.nan)
    # 计算每行（样本）与每列（SNP）的缺失率
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
    按缺失率阈值过滤样本与 SNP（默认阈值较宽松，可在后续调整）。

    :param X: 基因型矩阵。
    :param sample_missing: 每个样本的缺失率 (pd.Series)。
    :param snp_missing: 每个 SNP 的缺失率 (pd.Series)。
    :param max_sample_missing: 样本级最大缺失率阈值（默认 0.8）。
    :param max_snp_missing: SNP 级最大缺失率阈值（默认 0.8）。
    :return: 过滤后的矩阵 (pd.DataFrame)，索引从 0 重新开始。
    """
    keep_rows = sample_missing <= max_sample_missing
    keep_cols = snp_missing[snp_missing <= max_snp_missing].index
    return X.loc[keep_rows, keep_cols].reset_index(drop=True)


def _fill_mode(Z: pd.DataFrame, fallback: float = 1) -> pd.DataFrame:
    """
    列众数填补（默认方法）。

    :param Z: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param fallback: 当整列均为空时的回退值（默认填 1）。
    :return: 填补后的矩阵 (pd.DataFrame)。
    说明:
        - 对每列单独计算众数；
        - 若整列无众数则使用 fallback；
        - 输出与原矩阵结构保持一致。
    """

    def fill_mode(col: pd.Series) -> pd.Series:
        """
        对单列执行众数填补。

        :param col: 一列 SNP 数据（可能包含 NaN）。
        :return: 填补后的该列。
        说明:
            - 使用 value_counts() 统计各值频率；
            - 若整列为空则使用外层 fallback 值；
            - 返回与输入列等长的 Series。
        """
        vc = col.dropna().value_counts()               # 统计每个值出现次数
        mode_value = vc.index[0] if len(vc) else fallback  # 取众数或回退值
        return col.fillna(mode_value)                  # 用众数填补缺失值

    return Z.apply(fill_mode, axis=0)  # 按列执行 fill_mode


def _fill_mean(X: pd.DataFrame) -> pd.DataFrame:
    """
    列均值填补。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :return: 填补后的矩阵。
    说明:
        - 对每列单独计算均值；
        - 用 fillna() 替换 NaN；
        - 输出与原矩阵列名一致。
    """
    return X.fillna(X.mean())


def _fill_knn(X: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    KNN 填补（基于样本相似度）。

    :param X: 基因型矩阵 (pd.DataFrame)。
    :param n_neighbors: 近邻数（默认 5）。
    :return: 填补后的矩阵。
    说明:
        - 使用 sklearn.impute.KNNImputer；
        - metric='nan_euclidean' 表示计算距离时自动忽略 NaN；
        - 每个样本缺失值由最相似样本的均值替代。
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, metric="nan_euclidean")
    M = imputer.fit_transform(X)
    return pd.DataFrame(M, columns=X.columns)


def impute_missing(X: pd.DataFrame, method: str = "mode", n_neighbors: int = 5) -> pd.DataFrame:
    """
    缺失值填补。

    :param X: 基因型矩阵 (pd.DataFrame)。（编码规则: 0 = 参考等位基因, 1 = 变异等位基因, 3 = 缺失）
    :param method: 填补方法（'mode' / 'mean' / 'knn'）。
    :param n_neighbors: KNN 填补时的近邻数（默认 5）。
    :return: 填补后的矩阵 (pd.DataFrame)。
        说明:
        - mode: 每列取众数；
        - mean: 每列取均值；
        - knn : 基于样本相似度插补；
        - 所有方法接口一致：接收 DataFrame，返回 DataFrame。
    """
    Z = X.replace(3, np.nan)
    if method == "mode":
        return _fill_mode(Z)
    if method == "mean":
        return _fill_mean(Z)
    if method == "knn":
        return _fill_knn(Z, n_neighbors=n_neighbors)
    raise ValueError(f"未知填补方法: {method}")
