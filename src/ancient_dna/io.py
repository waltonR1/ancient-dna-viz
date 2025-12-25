"""
io.py
=====

文件读写模块（I/O Layer）
Input/Output module for file read and write operations.

本模块用于加载与保存基因型矩阵、样本注释表以及通用 CSV 文件，
是整个分析管线中的数据入口与出口组件。

This module provides utilities for loading and saving genotype
matrices, sample metadata tables, and general CSV files.
It serves as the data input/output layer of the analysis pipeline.

Functions
---------
load_geno
    读取基因型矩阵（CSV），提取样本 ID、SNP 数值矩阵及 SNP 列名。
    Read a genotype matrix (CSV) and extract sample IDs,
    the SNP matrix, and SNP column names.

load_meta
    读取样本注释表（CSV），并规范化样本 ID。
    Read a sample metadata table and normalize sample IDs.

load_csv
    通用 CSV 加载函数，包含基础错误处理与信息提示。
    General CSV loading utility with error handling and logging.

save_csv
    将任意 DataFrame 安全导出为 CSV 文件。
    Export any DataFrame to a CSV file with optional logging.

Description
-----------
本模块是整个管线的“数据 I/O 层（Data I/O Layer）”，
负责数据文件的读取、基础清洗、结构化转换以及安全保存。
所有上层模块（如 preprocess、embedding、clustering 等）
均依赖此模块提供的标准化数据结构接口。

This module serves as the data input/output layer of the pipeline.
It handles data loading, basic cleaning, structural conversion,
and safe persistence. All higher-level modules (e.g. preprocess,
embedding, clustering) rely on this module for standardized
data access interfaces.
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd


def load_geno(
        path: str | Path,
        id_col: str = "Genetic ID",
        sep: str = ";"
) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
    读取基因型矩阵文件并提取样本 ID 与 SNP 矩阵

    Load a genotype matrix and extract sample IDs and SNP data.

    本函数从 CSV 文件中读取基因型矩阵，
    提取样本 ID 列与对应的 SNP 特征列，
    并将 SNP 数据转换为数值矩阵以供后续分析使用。

    This function reads a genotype matrix from a CSV file,
    extracts the sample ID column and SNP feature columns,
    and converts SNP values into a numeric matrix for
    downstream analysis.

    Parameters
    ----------
    path : str or pathlib.Path
        基因型矩阵文件路径（CSV）。
        文件中应包含一列样本 ID，其余列为 SNP 特征。

        Path to the genotype CSV file. The file must contain
        a sample ID column, and the remaining columns represent SNPs.

    id_col : str, default="Genetic ID"
        样本 ID 所在的列名。

        Name of the column containing sample IDs.

    sep : str, default=";"
        CSV 文件的分隔符。

        Delimiter used in the CSV file.

    Returns
    -------
    ids : pandas.Series
        样本 ID 序列，与输入文件中的样本顺序一致。

        Series of sample IDs, preserving the original order.

    X : pandas.DataFrame
        SNP 数值矩阵，行表示样本，列表示 SNP 特征。
        无法解析为数值的元素将被置为 NaN。

        Numeric SNP matrix with samples as rows and SNPs as columns.
        Unparseable values are converted to NaN.

    snp_cols : list of str
        SNP 特征列名列表（不包含样本 ID 列）。

        List of SNP column names (excluding the sample ID column).

    Notes
    -----
    - 本函数会自动去除列名与样本 ID 中的首尾空白字符。

      Column names and sample IDs are stripped of leading
      and trailing whitespace.

    - 除样本 ID 列外，其余列将统一尝试转换为数值类型。

      All columns except the sample ID column are coerced
      to numeric values.

    - 若指定的样本 ID 列不存在，将抛出 ``KeyError``。

      A ``KeyError`` is raised if the specified sample ID
      column is not found.
    """

    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if id_col not in df.columns:
        raise KeyError(f"Sample ID column not found: {id_col}")
    ids = df[id_col].astype(str).str.strip()
    snp_cols = [c for c in df.columns if c != id_col]
    X = df[snp_cols].apply(pd.to_numeric, errors="coerce")
    return ids, X, snp_cols


def load_meta(
        path: str | Path,
        id_col: str = "Genetic ID",
        sep: str = ";"
) -> pd.DataFrame:
    """
    读取样本注释表并规范化样本 ID

    Load a sample metadata table and normalize sample IDs.

    本函数从 CSV 文件中读取样本注释表（metadata），
    要求文件中包含样本 ID 列，并对样本 ID 进行
    字符串化与去除首尾空白的规范化处理。

    This function reads a sample metadata table from a CSV file,
    requires the presence of a sample ID column, and normalizes
    sample IDs by converting them to strings and stripping
    leading/trailing whitespace.

    Parameters
    ----------
    path : str or pathlib.Path
        样本注释表文件路径（CSV）。

        Path to the CSV file containing sample metadata.

    id_col : str, default="Genetic ID"
        样本 ID 所在的列名。

        Name of the column containing sample IDs.

    sep : str, default=";"
        CSV 文件的分隔符。

        Delimiter used in the CSV file.

    Returns
    -------
    pandas.DataFrame
        样本注释表，包含规范化后的样本 ID 列，
        其余列保持原样。

        Sample metadata table with normalized sample IDs.

    Notes
    -----
    - 本函数会自动去除所有列名中的首尾空白字符。

      All column names are stripped of leading and
      trailing whitespace.

    - 样本 ID 列会被显式转换为字符串类型。

      The sample ID column is explicitly converted
      to string type.

    - 若指定的样本 ID 列不存在，将抛出 ``KeyError``。

      A ``KeyError`` is raised if the specified sample
      ID column is missing.
    """

    meta = pd.read_csv(path, sep=sep, dtype=str)
    meta.columns = [c.strip() for c in meta.columns]
    if id_col not in meta.columns:
        raise KeyError(f"The annotation table is missing the Sample ID column.: {id_col}")
    meta[id_col] = meta[id_col].astype(str).str.strip()
    return meta


def load_csv(
        path: str | Path,
        sep: str = ";"
) -> pd.DataFrame:
    """
    通用 CSV 文件加载函数

    General-purpose CSV loading utility.

    本函数用于读取任意 CSV 文件，
    提供基础的路径检查、错误处理以及
    成功加载后的信息提示。

    This function loads a CSV file with basic path validation,
    error handling, and informative logging upon success.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV 文件路径。

        Path to the CSV file.

    sep : str, default=";"
        CSV 文件分隔符。

        Delimiter used in the CSV file.

    Returns
    -------
    pandas.DataFrame
        读取并解析后的 DataFrame。

        The loaded DataFrame.

    Notes
    -----
    - 在尝试读取文件之前，会先检查文件是否存在。

      The function checks whether the file exists
      before attempting to load it.

    - 成功读取后会打印文件名、行数和列数。

      Upon successful loading, the function prints
      the file name along with row and column counts.

    - 读取过程中出现的异常将被捕获并重新抛出为
      ``RuntimeError``，以提供更清晰的错误信息。

      Any exception raised during loading is caught
      and re-raised as a ``RuntimeError`` with context.

    Raises
    ------
    FileNotFoundError
        当指定的文件路径不存在时抛出。

        Raised when the specified file path does not exist.

    RuntimeError
        当 CSV 读取失败时抛出。

        Raised when loading the CSV file fails.
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


def save_csv(
        df: pd.DataFrame,
        path: str | Path,
        sep: str = ";",
        index: bool = False,
        verbose: bool = True,
) -> None:
    """
    将 DataFrame 导出为 CSV 文件

    Export a DataFrame to a CSV file.

    本函数将任意 pandas DataFrame 写入 CSV 文件，
    在必要时会自动创建父目录，并可选输出保存完成的信息。

    This function writes a pandas DataFrame to a CSV file,
    automatically creates parent directories if needed,
    and optionally prints a confirmation message.

    Parameters
    ----------
    df : pandas.DataFrame
        需要导出的 DataFrame（如 SNP 矩阵、样本注释表或合并后的数据集）。

        The DataFrame to export (e.g. SNP matrix, metadata table,
        or merged dataset).

    path : str or pathlib.Path
        输出 CSV 文件路径。

        Output path of the CSV file.

    sep : str, default=";"
        CSV 文件分隔符。

        Delimiter used in the CSV file.

    index : bool, default=False
        是否在 CSV 文件中写入 DataFrame 的行索引。

        Whether to write the DataFrame index to the CSV file.

    verbose : bool, default=True
        是否在保存完成后打印提示信息。

        Whether to print a confirmation message after saving.

    Returns
    -------
    None
        本函数不返回值，结果直接写入文件。

        This function returns ``None`` and writes the result
        directly to disk.

    Notes
    -----
    - 若输出路径的父目录不存在，将自动创建。

      Parent directories of the output path are created
      automatically if they do not exist.

    - CSV 文件将使用指定的分隔符写入。

      The CSV file is written using the specified delimiter.

    - 当 ``verbose=True`` 时，将打印保存后的文件路径
      及数据行数信息。

      When ``verbose=True``, the function prints the saved
      file path and the number of rows written.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,sep = sep, index = index)

    if verbose:
        print(f"[OK] Saved Dataset: {path.resolve()} ({len(df)} rows)")