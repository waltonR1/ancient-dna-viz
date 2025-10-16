from pathlib import Path
from typing import Tuple, List
import pandas as pd

def load_geno(path: str | Path, id_col: str = "Genetic ID", sep: str = ";") -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
    读取基因型矩阵（CSV；默认分号分隔），包含样本ID列与若干SNP列。

    :param path: 文件路径（CSV）。首列/指定列应为样本ID，其余列为SNP（如rsID）。
    :param id_col: 样本ID列名（默认 "Genetic ID"）。
    :param sep: CSV分隔符（默认 ";"）。
    :return: (ids, X, snp_cols)
             - ids: 样本ID序列 (pd.Series)
             - X: SNP数值矩阵 (pd.DataFrame)，行=样本，列=SNP
             - snp_cols: SNP列名列表
    说明:
        - 除ID列外的所有列将尝试转为数值；无法解析的值会变为 NaN。
    """
    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if id_col not in df.columns:
        raise KeyError(f"未找到样本 ID 列: {id_col}")
    ids = df[id_col].astype(str).str.strip()
    snp_cols = [c for c in df.columns if c != id_col]
    X = df[snp_cols].apply(pd.to_numeric, errors="coerce")
    return ids, X, snp_cols


def load_meta(path: str | Path, id_col: str = "Genetic ID", sep: str = ";") -> pd.DataFrame:
    """
    读取样本注释表（CSV），通常包含 Y/mt 单倍群、World Zone、位置、性别等。

    :param path: 文件路径（CSV），需包含样本ID列。
    :param id_col: 样本ID列名（默认 "Genetic ID"）。
    :param sep: CSV分隔符（默认 ";"）。
    :return: meta (pd.DataFrame)
    """
    meta = pd.read_csv(path, sep=sep, dtype=str)
    meta.columns = [c.strip() for c in meta.columns]
    if id_col not in meta.columns:
        raise KeyError(f"注释表缺少样本 ID 列: {id_col}")
    meta[id_col] = meta[id_col].astype(str).str.strip()
    return meta

def load_csv(path: str | Path, sep: str = ";") -> pd.DataFrame:
    """
    通用 CSV 加载函数。

    :param path: 文件路径。
    :param sep: 分隔符（默认 ; ，兼容内部标准）。
    :return: 读取的 DataFrame。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep=sep)
    print(f"[OK] Loaded CSV: {path.name} ({len(df)} rows, {len(df.columns)} cols)")
    return df

def save_csv(df: pd.DataFrame, path: str | Path, sep: str = ";", index: bool = False) -> None:
    """
    导出任意 DataFrame 为 CSV 文件。

    :param df: 需要导出的 DataFrame（可为SNP矩阵或合并后的大表）。
    :param path: 输出文件路径。
    :param sep: 分隔符（默认 ","）。
    :param index: 是否导出 DataFrame 的行索引（默认 False）。
    :return: None（直接写入文件）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,sep = sep, index = index)
    print(f"[OK] 已导出 → {path.resolve()}")

def load_table(path: str | Path, sep: str = ";") -> pd.DataFrame:
    """
    通用表格加载函数（CSV / TSV）。

    :param path: 文件路径。
    :param sep: 分隔符（默认 ";"）。
    :return: 加载后的 DataFrame。
    说明:
        - 用于读取任意中间结果或统计报告；
        - 自动检测 UTF-8；
        - 若无法解析，抛出详细异常。
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=sep)
        print(f"[OK] 已加载表格: {path}")
        return df
    except Exception as e:
        raise RuntimeError(f"无法加载表格 {path}: {e}")

def merge_matrix_meta(
    ids: pd.Series,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    id_col_meta: str = "Genetic ID",
    meta_first: bool = True
) -> pd.DataFrame:
    """
    合并基因型矩阵与注释表，生成“ID + 注释 + SNP”的宽表。

    :param ids: 样本ID序列（列名将统一为 "Genetic ID"）。
    :param X: SNP矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param meta: 注释表（至少包含样本ID列）。
    :param id_col_meta: 注释表中样本ID列名（默认 "Genetic ID"）。
    :param meta_first: 是否让注释列在左侧（True）或右侧（False）。
    :return: 合并后的宽表 DataFrame。
    """
    ids_df = pd.DataFrame({"Genetic ID": ids.astype(str).values})
    df = pd.concat([ids_df, X.reset_index(drop=True)], axis=1)
    if id_col_meta != "Genetic ID":
        meta = meta.rename(columns={id_col_meta: "Genetic ID"})
    if meta_first:
        merged = meta.merge(df, on="Genetic ID", how="left")
    else:
        merged = df.merge(meta, on="Genetic ID", how="left")
    return merged