"""
io.py
-----------------
文件读写（I/O）模块
I/O module for file read/write operations.

用于加载与保存基因型矩阵、样本注释表及通用 CSV 文件。
Used to load and save genotype matrices, sample metadata tables, and general CSV files.

功能：
    - load_geno(): 读取基因型矩阵（CSV），提取样本ID与SNP矩阵；
    - load_meta(): 读取样本注释表（含样本ID及元数据）；
    - load_csv(): 通用CSV加载函数（带错误处理与信息提示）；
    - save_csv(): 导出任意DataFrame为CSV文件。
Functions:
    - load_geno(): Read genotype matrix (CSV) and extract sample IDs and SNP matrix.
    - load_meta(): Read sample metadata table (includes sample IDs and metadata).
    - load_csv(): General CSV loading function with error handling and logging.
    - save_csv(): Export any DataFrame to a CSV file.

说明：
    本模块是整个管线的数据入口与出口组件（Data I/O Layer），
    负责数据文件的读取、清洗、转换与安全保存。
    所有上层模块（如 preprocess、embedding、clustering 等）
    均依赖此模块提供的标准化数据结构接口。
Description:
    This module serves as the data input/output layer for the entire pipeline.
    It is responsible for reading, cleaning, transforming, and safely saving data files.
    All higher-level modules (e.g., preprocess, embedding, clustering)
    rely on this module to provide standardized data structure interfaces.
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd

def load_geno(path: str | Path, id_col: str = "Genetic ID", sep: str = ";") -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
    读取基因型矩阵（CSV；默认分号分隔），包含样本ID列与若干SNP列。
    Load a genotype matrix (CSV; semicolon-separated by default) containing a sample ID column and multiple SNP columns.

    :param path:
        文件路径（CSV）。首列/指定列应为样本ID，其余列为SNP（如rsID）。
        Path to the CSV file. The first/specified column should contain sample IDs, and the remaining columns represent SNPs (e.g., rsIDs).

    :param id_col:
        样本ID列名（默认 "Genetic ID"）。
        Name of the sample ID column (default: "Genetic ID").

    :param sep:
        CSV分隔符（默认 ";"）。
        CSV delimiter (default: ";").

    :return:
        (ids, X, snp_cols)
        - ids: 样本ID序列 (pd.Series) / Series of sample IDs
        - X: SNP数值矩阵 (pd.DataFrame)，行=样本，列=SNP / Numeric SNP matrix (rows=samples, columns=SNPs)
        - snp_cols: SNP列名列表 / List of SNP column names

    说明 / Notes:
        - 除ID列外的所有列将尝试转为数值；无法解析的值会变为 NaN。
        - All columns except the ID column will be converted to numeric; unparseable values become NaN.
    """
    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if id_col not in df.columns:
        raise KeyError(f"Sample ID column not found: {id_col}")
    ids = df[id_col].astype(str).str.strip()
    snp_cols = [c for c in df.columns if c != id_col]
    X = df[snp_cols].apply(pd.to_numeric, errors="coerce")
    return ids, X, snp_cols


def load_meta(path: str | Path, id_col: str = "Genetic ID", sep: str = ";") -> pd.DataFrame:
    """
    读取样本注释表（CSV），需包含样本ID列。
    Load a sample metadata table (CSV) that includes a sample ID column.

    :param path:
        文件路径（CSV），需包含样本ID列。
        Path to the CSV file containing sample IDs.

    :param id_col:
        样本ID列名（默认 "Genetic ID"）。
        Name of the sample ID column (default: "Genetic ID").

    :param sep:
        CSV分隔符（默认 ";"）。
        CSV delimiter (default: ";").

    :return:
        meta (pd.DataFrame) 样本注释表。
        meta (pd.DataFrame) — Sample metadata table.
    """
    meta = pd.read_csv(path, sep=sep, dtype=str)
    meta.columns = [c.strip() for c in meta.columns]
    if id_col not in meta.columns:
        raise KeyError(f"The annotation table is missing the Sample ID column.: {id_col}")
    meta[id_col] = meta[id_col].astype(str).str.strip()
    return meta


def load_csv(path: str | Path, sep: str = ";") -> pd.DataFrame:
    """
    通用 CSV 加载函数（含错误处理与提示）。
    General CSV loading function (with error handling and informative messages).

    :param path:
        文件路径。
        Path to the file.

    :param sep:
        分隔符（默认 ";"，兼容内部标准）。
        Delimiter (default: ";" for internal consistency).

    :return:
        读取的 DataFrame。
        The loaded DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path, sep=sep)
        print(f"[OK] Loaded CSV: {path.name} ({len(df)} rows, {len(df.columns)} cols)")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV {path}: {e}")


def save_csv(df: pd.DataFrame, path: str | Path, sep: str = ";", index: bool = False, verbose: bool = True,) -> None:
    """
    导出任意 DataFrame 为 CSV 文件。
    Export any DataFrame to a CSV file.

    :param df:
        需要导出的 DataFrame（可为SNP矩阵或合并后的大表）。
        The DataFrame to export (can be SNP matrix or merged dataset).

    :param path:
        输出文件路径。
        Output file path.

    :param sep:
        分隔符（默认 ","）。
        Delimiter (default: ",").

    :param index:
        是否导出 DataFrame 的行索引（默认 False）。
        Whether to include the DataFrame index (default: False).

    :param verbose:
        是否打印保存信息（默认 True）。
        Whether to print save confirmation (default: True).

    :return:
        None（直接写入文件）。
        None (writes directly to file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,sep = sep, index = index)

    if verbose:
        print(f"[OK] Saved Dataset: {path.resolve()} ({len(df)} rows)")