from pathlib import Path
import ancient_dna as adna

def main():
    ROOT = Path(__file__).resolve().parents[1]
    geno_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct 1.csv"
    meta_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct_classes 1.csv"
    geno_proceed = ROOT / "data" / "processed" / "geno.csv"
    meta_proceed = ROOT / "data" / "processed" / "meta.csv"


    ids, X, _ = adna.load_geno(geno_path)
    meta = adna.load_meta(meta_path)

    X1, meta1 = adna.align_by_id(ids, X, meta)
    sm, cm = adna.compute_missing_rates(X1)
    Xf = adna.filter_by_missing(X1, sm, cm)
    Xi = adna.impute_missing(Xf)

    adna.save_csv(Xi, geno_proceed)
    adna.save_csv(meta1, meta_proceed)
    print("[OK] Saved:", geno_proceed, meta_proceed)

    proj = adna.project_embeddings(Xi, method="umap", n_components=2)
    adna.plot_embedding(proj, labels=meta1["haplogroup"] if "haplogroup" in meta1.columns else None,
                             title="UMAP Projection of Samples")

if __name__ == "__main__":
    main()
