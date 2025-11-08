"""
inspect_ancestrymap.py
-----------------
数据集检查模块（Data Inspection Layer）
Lightweight inspection tool for AncestryMap / EIGENSTRAT datasets (.geno / .snp / .ind / .anno).

用于快速查看并验证遗传数据集的结构与内容，
兼容二进制与文本格式的 .geno 文件，自动检测并解码。
Provides a quick inspection interface for dataset integrity checking,
automatically detects and decodes both binary and text-based .geno formats.

功能 / Functions:
    - inspect_geno(): 检查 GENO 文件结构与格式（自动识别文本/二进制）
      Inspect GENO file structure and format (auto-detect text/binary).
    - inspect_snp(): 解析 SNP 文件并显示染色体与位点统计。
      Parse SNP file and show chromosome/variant summaries.
    - inspect_ind(): 读取 IND 文件并显示性别与种群分布。
      Read IND file and summarize sex/population distribution.
    - inspect_anno(): 检查 ANNO 文件结构与列信息。
      Inspect ANNO annotation file and list column structure.
    - main(): 主入口函数，批量检查所有数据文件。
      Main entry point that runs all inspection steps sequentially.

说明 / Description:
    本模块属于数据处理流程的“检查层”（Inspection Layer），
    用于在加载与分析前，验证 AncestryMap 格式数据集的完整性与可读性。
    This module serves as the inspection layer of the pipeline,
    verifying the integrity and readability of AncestryMap-formatted datasets
    before preprocessing and downstream analysis.
"""

from pathlib import Path
import pandas as pd
from ancient_dna.genotopython import unpackfullgenofile


def print_sep(title: str):
    """打印分隔标题 / Print section separator."""
    print("\n" + "=" * 25)
    print(f"=== {title}")
    print("=" * 25)


def inspect_geno(geno_path: Path, max_lines: int = 5, max_cols: int = 100):
    """
    检查 GENO 文件（自动识别文本或二进制格式）
    Inspect GENO file and auto-detect whether it's text or binary.

    :param geno_path: Path
        GENO 文件路径 / Path to .geno file.
    :param max_lines: int
        预览的最大行数 / Number of preview lines for text mode.
    :param max_cols: int
        每行预览的最大字符数 / Max number of characters displayed per line.

    说明 / Notes:
        - 自动检测文件类型：ASCII 文本或二进制格式。
          Automatically detects whether file is text or binary encoded.
        - 文本模式下统计 SNP 行数与字符分布。
          Reports SNP rows and character frequencies for text mode.
        - 二进制模式下调用 unpackfullgenofile() 进行解码。
          Uses unpackfullgenofile() to decode binary .geno files.
    """
    print_sep("GENO FILE")
    if not geno_path.exists():
        print(f"[WARN] {geno_path.name} not found.")
        return

    # --- 检查是否是文本型 GENO ---
    with open(geno_path, "rb") as f:
        head = f.read(200)
    is_binary = head.startswith(b"GENO") or any(b > 127 for b in head)
    print(f"[INFO] Detected {'binary' if is_binary else 'text'} GENO format.")

    if not is_binary:
        # 文本格式 (.geno 为 ASCII 行)
        total_lines = 0
        n_cols = None
        char_counter = {}
        with open(geno_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if n_cols is None:
                    n_cols = len(line)
                total_lines += 1
                for c in line:
                    char_counter[c] = char_counter.get(c, 0) + 1
                if total_lines <= max_lines:
                    print(f"Line {total_lines:>3}: {line[:max_cols]}{'...' if len(line)>max_cols else ''}")

        print(f"\n[INFO] SNP rows: {total_lines:,} | Individuals per row: {n_cols:,}")
        print(f"[INFO] Approx. matrix size: {total_lines * n_cols / 1e6:.2f} million genotypes")
        print(f"[INFO] Value frequency (top 5): {dict(sorted(char_counter.items())[:5])}")
        return

    # --- 二进制格式 GENO ---
    try:
        geno, nind, nsnp, rlen = unpackfullgenofile(geno_path)

        print(f"[OK] Parsed binary GENO: {nsnp:,} SNPs × {nind:,} individuals")
        print(f"[INFO] Raw byte length: {len(geno):,}")
        print(f"[INFO] Header size: {rlen} bytes")

        print("[INFO] Displaying first few raw bytes (preview):")
        print(geno[:100])

        print(f"\n[INFO] (Binary packed GENO successfully decoded.)")

    except Exception as e:
        print(f"[ERROR] Could not decode binary GENO: {e}")


def inspect_snp(snp_path: Path, n_rows: int = 5):
    """
    检查 SNP 文件
    Inspect SNP file and summarize basic variant information.

    :param snp_path: Path
        SNP 文件路径 / Path to .snp file.
    :param n_rows: int
        预览行数 / Number of rows to preview.

    说明 / Notes:
        - 解析常见 6 列格式（SNP_ID, CHR, GEN_DIST, BP_POS, ALLELE1, ALLELE2）。
          Reads standard 6-column EIGENSTRAT SNP file.
        - 打印行数与染色体分布前十名。
          Prints basic statistics and top chromosome counts.
    """
    print_sep("SNP FILE")
    if not snp_path.exists():
        print(f"[WARN] {snp_path.name} not found.")
        return

    df = pd.read_csv(
        snp_path, sep=r"\s+", header=None,
        names=["SNP_ID", "CHR", "GEN_DIST", "BP_POS", "ALLELE1", "ALLELE2"]
    )
    print(f"[INFO] SNPs: {len(df):,}")
    print(df.head(n_rows))
    print("\n[INFO] Top chromosomes:")
    print(df["CHR"].value_counts().head(10))


def inspect_ind(ind_path: Path, n_rows: int = 10):
    """
    检查 IND 文件
    Inspect IND file and summarize individuals' attributes.

    :param ind_path: Path
        IND 文件路径 / Path to .ind file.
    :param n_rows: int
        预览行数 / Number of rows to preview.

    说明 / Notes:
        - 读取标准三列格式（Individual, Sex, Population）。
          Reads standard 3-column IND file.
        - 统计性别比例与种群分布前十。
          Displays sex ratio and top 10 population counts.
    """
    print_sep("IND FILE")
    if not ind_path.exists():
        print(f"[WARN] {ind_path.name} not found.")
        return

    df = pd.read_csv(ind_path, sep=r"\s+", header=None,
                     names=["Individual", "Sex", "Population"])
    print(f"[INFO] Individuals: {len(df):,}")
    print(df.head(n_rows))
    print("\n[INFO] Sex distribution:")
    print(df["Sex"].value_counts())
    print("\n[INFO] Top populations:")
    print(df["Population"].value_counts().head(10))


def inspect_anno(anno_path: Path, n_rows: int = 10):
    """
    检查 ANNO 文件
    Inspect annotation (.anno) file.

    :param anno_path: Path
        注释文件路径 / Path to annotation file.
    :param n_rows: int
        预览行数 / Number of rows to preview.

    说明 / Notes:
        - 支持自动检测分隔符（Tab / Comma）。
          Automatically detects separator (Tab or Comma).
        - 打印列名与前几行数据预览。
          Displays column names and top rows.
        - 可用于检查样本标签与地理信息是否完整。
          Useful for verifying metadata completeness.
    """
    print_sep("ANNO FILE")
    if not anno_path.exists():
        print(f"[WARN] {anno_path.name} not found.")
        return

    try:
        df = pd.read_csv(anno_path, sep="\t", header=0, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"[WARN] Tab read failed ({e}); retrying with comma separator...")
        df = pd.read_csv(anno_path, sep=",", header=0, engine="python", on_bad_lines="skip")

    print(f"[INFO] Annotation: {df.shape[0]:,} samples × {df.shape[1]} columns")
    print(df.head(n_rows))
    print("\n[INFO] Columns:", list(df.columns))


def main():
    """
    主入口函数
    Main entry point for dataset inspection.

    自动执行对 .geno / .snp / .ind / .anno 文件的检测与报告。
    Automatically performs inspection and summary of all data files.

    :return: None
        无返回值，直接打印检查结果。
        None (prints results to console).
    """
    ROOT = Path(__file__).resolve().parents[1] if __file__.endswith(".py") else Path.cwd()
    raw_dir = ROOT / "data" / "raw"

    prefix = "v54.1.p1_HO_public"
    print(f"[INFO] Inspecting AncestryMap dataset: {prefix}")

    geno_path = raw_dir / f"{prefix}.geno"
    snp_path = raw_dir / f"{prefix}.snp"
    ind_path = raw_dir / f"{prefix}.ind"
    anno_path = raw_dir / f"{prefix}.anno"

    inspect_geno(geno_path)
    inspect_snp(snp_path)
    inspect_ind(ind_path)
    inspect_anno(anno_path)

    print_sep("DONE")
    print("[OK] Dataset inspection completed.")


if __name__ == "__main__":
    main()
