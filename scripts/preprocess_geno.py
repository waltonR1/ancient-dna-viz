import argparse
import ancient_dna as adna
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_matrix", default="clean_matrix.csv")
    ap.add_argument("--out_meta", default="meta_aligned.csv")
    ap.add_argument("--zero_as_missing", action="store_true")
    args = ap.parse_args()

    ids, X, _ = adna.load_geno(args.geno)
    meta = adna.load_meta(args.meta)

    X1, meta1 = adna.align_by_id(ids, X, meta)
    sm, cm = adna.compute_missing_rates(X1, zero_as_missing=args.zero_as_missing)
    Xf = adna.filter_by_missing(X1, sm, cm)
    Xi = adna.impute_missing(Xf, zero_as_missing=args.zero_as_missing)

    Xi.to_csv(args.out_matrix, index=False)
    meta1.to_csv(args.out_meta, index=False)
    print("[OK] Saved:", args.out_matrix, args.out_meta)

if __name__ == "__main__":
    main()
