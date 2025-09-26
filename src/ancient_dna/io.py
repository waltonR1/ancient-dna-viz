from pathlib import Path
from typing import Tuple, List

import numpy as np
import gzip
from math import ceil
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


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """
    导出任意 DataFrame 为 CSV 文件。

    :param df: 需要导出的 DataFrame（可为SNP矩阵或合并后的大表）。
    :param path: 输出文件路径。
    :param index: 是否导出 DataFrame 的行索引（默认 False）。
    :return: None（直接写入文件）。
    """
    path = Path(path)
    df.to_csv(path, index=index)
    print(f"[OK] 已导出 → {path.resolve()}")

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



def read_ind(path: str | Path) -> pd.DataFrame:
    """
    读取 .ind（个体信息）文件，常见列：个体ID、性别(M/F/U)、群体标签。

    :param path: .ind 文件路径（纯文本；空白分隔）。
    :return: DataFrame，列名: ["indiv_id", "sex", "population"]
    """
    ind = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if ind.shape[1] < 3:
        raise ValueError(".ind 文件列数不足，期望至少3列：id, sex, population")
    ind = ind.iloc[:, :3].copy()
    ind.columns = ["indiv_id", "sex", "population"]
    ind["indiv_id"] = ind["indiv_id"].astype(str)
    return ind

def read_snp(path: str | Path) -> pd.DataFrame:
    """
    读取 .snp（位点信息）文件，常见6列：SNP名、染色体、遗传距离、物理位置、等位基因1、等位基因2。

    :param path: .snp 文件路径（纯文本；空白分隔）。
    :return: DataFrame，列名: ["snp_id","chrom","gen_dist","pos","allele1","allele2"]
    """
    snp = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if snp.shape[1] < 6:
        raise ValueError(".snp 文件列数不足，期望至少6列")
    snp = snp.iloc[:, :6].copy()
    snp.columns = ["snp_id", "chrom", "gen_dist", "pos", "allele1", "allele2"]
    snp["snp_id"] = snp["snp_id"].astype(str)
    return snp

def read_anno(path: str | Path) -> pd.DataFrame:
    """
    读取 .anno（附加注释）文件。
    不同数据集格式可能不同，通常为空白或制表符分隔。此处做“尽量读入”的兼容处理。

    :param path: .anno 文件路径（纯文本；一般为空白/制表符分隔）。
    :return: DataFrame（不强制列名；尽量保留原始表头）
    """
    try:
        # 优先尝试制表符
        anno = pd.read_csv(path, sep="\t", dtype=str, comment="#")
    except Exception:
        # 回退为空白分隔
        anno = pd.read_csv(path, delim_whitespace=True, dtype=str, header=0, comment="#")
    anno.columns = [str(c).strip() for c in anno.columns]
    return anno

def _open_bin_auto(path: Path):
    """自动识别是否 gzip；以二进制方式打开。"""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":   # gzip magic
        return gzip.open(path, "rb")
    return open(path, "rb")


def _read_geno_text_or_gz(path: Path, missing_code: int | str = 9, dtype=np.int8) -> np.ndarray:
    """文本版 .geno（含 .gz）解析：每行一条SNP，字符仅 0/1/2/9。"""
    rows = []
    with _open_bin_auto(path) as f:
        for raw in f:
            raw = raw.rstrip(b"\r\n")
            if not raw:
                continue
            # 容错：去掉行内空格/Tab
            if b" " in raw or b"\t" in raw:
                raw = raw.replace(b" ", b"").replace(b"\t", b"")
            # 仅允许 '0','1','2','9'
            invalid = set(raw) - {48, 49, 50, 57}
            if invalid:
                raise ValueError(
                    f"检测到非文本 .geno（存在非 0/1/2/9 字节 {invalid}）。"
                    f" 如需自动解析 packed 格式，请使用 read_geno(..., expected_nind=..., expected_nsnps=...)。"
                )
            arr = (np.frombuffer(raw, dtype=np.uint8) - 48).astype(dtype, copy=False)
            rows.append(arr)
    if not rows:
        raise ValueError(f".geno 文件为空: {path}")
    mat = np.vstack(rows)
    miss_val = int(missing_code)
    mat[mat == miss_val] = -1
    return mat


def _read_geno_packed(path: Path, expected_nind: int, expected_nsnps: int,
                      missing_code_packed: int = 3, dtype=np.int8) -> np.ndarray:
    """
    解析 packed .geno（文件头以 b'GENO' 开始；每个基因型 2 bit 编码）。
    假设每个 SNP 连续存储，长度 = ceil(n_ind/4) 字节；2 bit 映射:
        0b00 -> 0, 0b01 -> 1, 0b10 -> 2, 0b11 -> 3(缺失)

    :param expected_nind: 个体数（来自 .ind 行数）
    :param expected_nsnps: SNP 数（来自 .snp 行数）
    :param missing_code_packed: 2bit 中的缺失码（通常为 3）
    """
    bytes_per_snp = ceil(expected_nind / 4)
    with open(path, "rb") as f:
        header = f.read(4)
        if header != b"GENO":
            raise ValueError("不是 packed .geno：缺少 'GENO' 文件头")
        data = f.read()
    needed = expected_nsnps * bytes_per_snp
    if len(data) < needed:
        raise ValueError(
            f"packed .geno 大小不匹配：需要 {needed} 字节（{expected_nsnps}×{bytes_per_snp}），实际 {len(data)}"
        )
    # 切成 (SNP, bytes_per_snp)
    buf = np.frombuffer(memoryview(data)[:needed], dtype=np.uint8)
    buf = buf.reshape((expected_nsnps, bytes_per_snp))

    # 解包：每字节 4 个 2bit 值（低位在前）
    out = np.empty((expected_nsnps, expected_nind), dtype=dtype)
    for i in range(expected_nsnps):
        b = buf[i]
        # 生成 4 列：00,01,10,11
        g0 = (b >> 0) & 0b11
        g1 = (b >> 2) & 0b11
        g2 = (b >> 4) & 0b11
        g3 = (b >> 6) & 0b11
        # 拼成 (bytes_per_snp*4,) 然后截断到 n_ind
        row = np.vstack([g0, g1, g2, g3]).T.reshape(-1)[:expected_nind]
        out[i, :] = row

    # 缺失值改为 -1
    out[out == missing_code_packed] = -1
    return out

def read_geno(path: str | Path,
              missing_code: int | str = 9,
              dtype=np.int8,
              expected_nind: int | None = None,
              expected_nsnps: int | None = None) -> np.ndarray:
    """
    通用 .geno 读取器：自动支持“文本 .geno / gzip 文本 .geno / packed .geno”。

    - 若文件前 4 字节为 b'GENO'，则按 packed 2bit 格式解析（需提供 expected_nind 与 expected_nsnps）。
    - 否则尝试按“文本 .geno”（含 .gz）解析（只允许 0/1/2/9）。

    :param path: .geno/.geno.gz 文件路径
    :param missing_code: 文本 .geno 的缺失码（默认 9；packed 使用 2bit 缺失=3）
    :param dtype: 输出矩阵 dtype（默认 np.int8）
    :param expected_nind: （仅 packed 需要）个体数，通常 = .ind 行数
    :param expected_nsnps:（仅 packed 需要）SNP 数，通常 = .snp 行数
    :return: numpy.ndarray，形状 = (num_snps, num_individuals)
    """
    path = Path(path)
    # 先看是不是 packed
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"GENO":
        if expected_nind is None or expected_nsnps is None:
            raise ValueError("检测到 packed .geno，但未提供 expected_nind/expected_nsnps 用于解包。")
        return _read_geno_packed(path, expected_nind, expected_nsnps, missing_code_packed=3, dtype=dtype)
    # 否则当作文本/文本.gz
    return _read_geno_text_or_gz(path, missing_code=missing_code, dtype=dtype)

def eigenstrat_to_dataframe(
    geno_mat: np.ndarray,
    snp: pd.DataFrame,
    ind: pd.DataFrame,
    missing_code: int = 9,
    missing_as: float | None = np.nan
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    将 .geno/.snp/.ind 组合成“行=个体、列=SNP”的 DataFrame（便于与你现有流程对接）。

    :param geno_mat: 从 read_geno 得到的矩阵，形状 (num_snps, num_individuals)。
    :param snp: 从 read_snp 得到的 DataFrame（需含列 'snp_id'）。
    :param ind: 从 read_ind 得到的 DataFrame（需含列 'indiv_id'）。
    :param missing_code: 缺失编码（默认 9）。
    :param missing_as: 输出 DataFrame 中缺失值的表示（默认 np.nan；可设为 -1 等）。
    :return: (X_df, ids, snp_cols)
             - X_df: 行=个体、列=SNP 的 DataFrame
             - ids: 个体ID (pd.Series)，与 X_df 行顺序一致
             - snp_cols: 列名列表（SNP顺序与 .snp 一致）
    说明:
        - EIGENSTRAT 的 .geno 每行是SNP、每列是个体，所以这里先转置。
        - 缺失码(9)会替换为 missing_as（默认 np.nan）。
    """
    if geno_mat.shape[0] != len(snp):
        raise ValueError(f".geno 行数({geno_mat.shape[0]}) 与 .snp 行数({len(snp)}) 不一致")
    # 转置为 (IND × SNP)
    M = geno_mat.T.astype(float)  # 先转 float，便于写入 NaN
    M[M == float(missing_code)] = np.nan if missing_as is np.nan else missing_as

    ids = ind["indiv_id"].astype(str).reset_index(drop=True)
    snp_cols = snp["snp_id"].astype(str).tolist()
    X_df = pd.DataFrame(M, index=ids, columns=snp_cols).reset_index()
    X_df = X_df.rename(columns={"index": "Genetic ID"})  # 与CSV流程统一命名
    ids_series = X_df["Genetic ID"]
    X_only = X_df.drop(columns=["Genetic ID"])
    return X_only, ids_series, snp_cols

def load_eigenstrat(
    prefix: str | Path,
    anno_path: str | Path | None = None,
    missing_code: int = 9,
    missing_as: float | None = np.nan,
    id_col_meta: str = "Genetic ID",
) -> Tuple[pd.Series, pd.DataFrame, List[str], pd.DataFrame | None]:
    """
    读取 EIGENSTRAT 四件套（或三件套），自动支持文本/packed .geno。

    :return: (ids, X, snp_cols, anno)
    """
    prefix = Path(prefix)
    geno_path = Path(str(prefix) + ".geno")
    snp_path  = Path(str(prefix) + ".snp")
    ind_path  = Path(str(prefix) + ".ind")

    snp = read_snp(snp_path)
    ind = read_ind(ind_path)

    # 关键：把个体数与SNP数传给 read_geno，支持 packed 解包
    G = read_geno(
        geno_path,
        missing_code=missing_code,
        dtype=np.int16,
        expected_nind=len(ind),
        expected_nsnps=len(snp),
    )

    X, ids, snp_cols = eigenstrat_to_dataframe(G, snp, ind, missing_code=9, missing_as=missing_as)
    anno = read_anno(anno_path) if anno_path is not None else None
    return ids, X, snp_cols, anno


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    try:
        geno_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct 1.csv"
        meta_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct_classes 1.csv"
        ids, X, snp_cols = load_geno(geno_path)
        meta = load_meta(meta_path)
        merged_csv = merge_matrix_meta(ids, X, meta)
        save_csv(merged_csv, ROOT / "data" / "processed" / "merged_csv.csv")
    except Exception as e:
        print("[CSV demo skipped]", e)

    # --- EIGENSTRAT 示例 ---
    # try:
    #     prefix = ROOT / "data" / "raw" / "v54.1.p1_HO_public"  # 不带后缀
    #     print(prefix)
    #     anno_path = str(prefix) + ".anno"
    #     ids2, X2, snps2, anno2 = load_eigenstrat(prefix, anno_path=anno_path, missing_code=9, missing_as=np.nan)
    #     # 如果有与 anno 对应的样本ID列名不同，可在 merge_matrix_meta 中改 id_col_meta
    #     merged_eig = merge_matrix_meta(ids2, X2, anno2 if anno2 is not None else pd.DataFrame({"Genetic ID": ids2}),
    #                                    id_col_meta="Genetic ID", meta_first=True)
    #     save_csv(merged_eig, ROOT / "data" / "processed" / "merged_eigenstrat.csv")
    # except Exception as e:
    #     print("[EIGENSTRAT demo skipped]", e)
