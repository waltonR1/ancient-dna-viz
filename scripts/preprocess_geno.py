import time
from pathlib import Path
import ancient_dna as adna

def main():
    ROOT = Path(__file__).resolve().parents[1]
    geno_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct 1.csv"
    meta_path = ROOT / "data" / "raw" / "Yhaplfiltered40pct_classes 1.csv"
    geno_proceed = ROOT / "data" / "processed" / "geno.csv"
    meta_proceed = ROOT / "data" / "processed" / "meta.csv"
    results_dir = ROOT / "data" / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

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
    labels = meta1["Y haplogroup"] if "Y haplogroup" in meta1.columns else None

    for method in methods:
        start = time.time()
        proj = adna.project_embeddings(Xi, method=method, n_components=2, random_state=42)
        elapsed = time.time() - start
        print(f"{method.upper()} finished in {elapsed:.2f} s")

        # 保存投影结果 CSV
        proj_path = results_dir / f"proj_{method}.csv"
        adna.save_csv(proj, proj_path)

        # 绘制并保存单独图像
        fig_path = results_dir / f"proj_{method}.png"
        adna.plot_embedding(
            proj,
            labels=labels,
            title=f"{method.upper()} Projection",
            save_path=fig_path
        )

    print(f"[OK] 投影结果和图像已保存到 {results_dir}")


if __name__ == "__main__":
    main()
