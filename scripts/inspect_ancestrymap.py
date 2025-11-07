"""
inspect_ancestrymap.py
轻量查看 AncestryMap / EIGENSTRAT 格式数据集 (.geno / .snp / .ind / .anno)
兼容二进制和文本格式的 .geno 文件，自动检测并读取。
"""

from pathlib import Path
import pandas as pd
from ancient_dna.genotopython import unpackfullgenofile


# ======================
# 辅助函数
# ======================
def print_sep(title: str):
    print("\n" + "=" * 25)
    print(f"=== {title}")
    print("=" * 25)


# ======================
# 1. GENO 文件
# ======================
def inspect_geno(geno_path: Path, max_lines: int = 5, max_cols: int = 100):
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


# ======================
# 2. SNP 文件
# ======================
def inspect_snp(snp_path: Path, n_rows: int = 5):
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


# ======================
# 3. IND 文件
# ======================
def inspect_ind(ind_path: Path, n_rows: int = 10):
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


# ======================
# 4. ANNO 文件
# ======================
def inspect_anno(anno_path: Path, n_rows: int = 10):
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


# ======================
# 主入口
# ======================
def main():
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
