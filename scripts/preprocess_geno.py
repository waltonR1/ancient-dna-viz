from pathlib import Path
import ancient_dna as adna

def main():
    ROOT = Path(__file__).resolve().parents[1]
    geno_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct 1.csv"
    meta_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct_classes 1.csv"
    geno_proceed = ROOT / "data" / "processed" / "geno.csv"
    meta_proceed = ROOT / "data" / "processed" / "meta.csv"
    out_path = ROOT / "data" / "results" / "compare_projections.png"

    # === 1. 加载数据 ===
    ids, X, _ = adna.load_geno(geno_path)
    meta = adna.load_meta(meta_path)

    # === 2. 对齐 & 预处理 ===
    X1, meta1 = adna.align_by_id(ids, X, meta)
    sm, cm = adna.compute_missing_rates(X1)
    Xf = adna.filter_by_missing(X1, sm, cm)
    Xi = adna.impute_missing(Xf)

    # === 3. 将预处理后的数据导出为csv ===
    adna.save_csv(Xi, geno_proceed)
    adna.save_csv(meta1, meta_proceed)
    print("[OK] Saved:", geno_proceed, meta_proceed)

    # === 4. 多方法对比 ===
    methods = ["umap", "tsne", "mds", "isomap"]

    # 使用库函数生成对比图（内部处理 plt）
    adna.plot_multiple_embeddings(
        X=Xi,
        meta=meta1,
        methods=methods,
        label_col="haplogroup",
        save_path=out_path,
        random_state=42
    )


if __name__ == "__main__":
    main()
