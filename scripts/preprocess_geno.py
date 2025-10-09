import time
from pathlib import Path

from scipy.sparse import SparseEfficiencyWarning

import ancient_dna as adna
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


def main():
    ROOT = Path(__file__).resolve().parents[1]
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "data" / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    # === 1. 加载数据 ===
    geno_path = raw_dir / "Yhaplfiltered40pct 1.csv"
    meta_path = raw_dir / "Yhaplfiltered40pct_classes 1.csv"
    print(f"[INFO] Loading data:\n- Geno: {geno_path}\n- Meta: {meta_path}")
    ids, X, _ = adna.load_geno(geno_path)
    meta = adna.load_meta(meta_path)

    # === 2. 对齐 & 预处理 ===
    print("[STEP] Aligning and computing missingness ...")
    X1, meta1 = adna.align_by_id(ids, X, meta)
    sm, cm = adna.compute_missing_rates(X1)
    Xf = adna.filter_by_missing(X1, sm, cm)
    # === 3. 生成缺失率报告 ===
    missing_report = adna.build_missing_report(sm, cm)
    report_path = results_dir / "missing_report.csv"
    adna.save_report(missing_report, report_path)
    print(f"[OK] Missingness report saved: {report_path.name}")

    # === 4. 定义方法 ===
    impute_methods = ["mode", "mean", "knn"]
    reduce_methods = ["umap", "tsne", "mds", "isomap"]
    labels = meta1["Y haplogroup"] if "Y haplogroup" in meta1.columns else None

    runtime_records = []

    # === 5. 主循环：多种缺失填补方法 ===
    for impute_method in impute_methods:
        print(f"\n[STEP] Imputation method: {impute_method.upper()}")

        # 5.1 填补缺失值
        Xi = adna.impute_missing(Xf, method=impute_method)
        geno_out = processed_dir / f"geno_{impute_method}.csv"
        meta_out = processed_dir / f"meta_{impute_method}.csv"
        adna.save_csv(Xi, geno_out)
        adna.save_csv(meta1, meta_out)
        print(f"[OK] Saved processed data for {impute_method}: {geno_out.name}, {meta_out.name}")

        # 5.2 对每种降维算法执行
        for method in reduce_methods:
            print(f"[RUN] {method.upper()} on {impute_method} ...")
            start = time.time()
            try:
                proj = adna.compute_embeddings(Xi, method=method, n_components=2, random_state=42)
                elapsed = time.time() - start
                print(f"{method.upper()} finished in {elapsed:.2f} s")

                # 记录运行时间
                runtime_records.append({
                    "imputation_method": impute_method,
                    "embedding_method": method,
                    "runtime_s": round(elapsed, 3)
                })

                # 保存降维结果与图像
                proj_path = results_dir / f"{impute_method}_proj_{method}.csv"
                adna.save_csv(proj, proj_path)

                fig_path = results_dir / f"{impute_method}_proj_{method}_Y_haplogroup.png"
                adna.plot_embedding(
                    proj,
                    labels=labels,
                    title=f"{method.upper()} ({impute_method}) Projection",
                    save_path=fig_path
                )

            except Exception as e:
                elapsed = time.time() - start
                print(f"[WARN] {method.upper()} failed on {impute_method}: {e}")
                runtime_records.append({
                    "imputation_method": impute_method,
                    "embedding_method": method,
                    "runtime_s": None,
                    "error": str(e)
                })
                continue

        print(f"[OK] All projections for {impute_method} saved to {results_dir}")

        # 5.3 为该填补方法生成报告
        embed_reports = {}
        for method in reduce_methods:
            path = results_dir / f"{impute_method}_proj_{method}.csv"
            if path.exists():
                emb = adna.load_csv(path)
                embed_reports[method] = adna.build_embedding_report(emb)

        if embed_reports:
            summary = adna.combine_reports(embed_reports)
            summary_path = results_dir / f"{impute_method}_embedding_summary.csv"
            adna.save_report(summary, summary_path)
            print(f"[OK] Embedding summary report saved: {summary_path.name}")

    # === 6. 导出运行时间报告 ===
    if runtime_records:
        runtime_path = results_dir / "runtime_summary.csv"
        adna.save_runtime_report(runtime_records, runtime_path)

    print("\n[ALL DONE] 所有缺失填补方法与降维结果已生成完毕！")
    print(f"[PATH] 查看结果目录: {results_dir}")


if __name__ == "__main__":
    main()