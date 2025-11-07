import time
import traceback
import warnings
from pathlib import Path
from scipy.sparse import SparseEfficiencyWarning
import ancient_dna as adna


# === 屏蔽冗余警告 ===
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)




# === 主函数 ===
def main():
    ROOT = Path(__file__).resolve().parents[1]
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    results_dir = ROOT / "data" / "results"
    for d in [raw_dir, processed_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === 1. 加载 Packed AncestryMap 数据 ===
    geno_path = raw_dir / "v54.1.p1_HO_public.geno"
    snp_path = raw_dir / "v54.1.p1_HO_public.snp"
    ind_path = raw_dir / "v54.1.p1_HO_public.ind"
    anno_path = raw_dir / "v54.1.p1_HO_public.anno"

    print(f"[INFO] Loading packed AncestryMap dataset:")
    print(f"- GENO: {geno_path}\n- SNP: {snp_path}\n- IND: {ind_path}\n- ANNO: {anno_path}")

    start_load = time.time()
    # 使用 ancient_dna 的接口（内部封装 genotopython）
    X = adna.genofileToPandas(str(geno_path), str(snp_path), str(ind_path), transpose=True)
    meta = adna.CreateLocalityFile(str(anno_path), toCSV=False, verbose=True)
    print(f"[OK] Data loaded in {time.time()-start_load:.2f}s | Matrix: {X.shape}, Meta: {meta.shape}")

    # === 2. 缺失值可视化与过滤 ===
    missing_plot_path = results_dir / "Missing_packed.png"
    adna.plot_missing_values(X, save_path=missing_plot_path)
    sm, cm = adna.compute_missing_rates(X)
    Xf = adna.filter_by_missing(X, sm, cm)
    adna.save_report(adna.build_missing_report(sm, cm), results_dir / "missing_report_packed.csv")

    adna.plot_missing_values(Xf, save_path=results_dir / "Missing_after_filtering_packed.png")

    # === 3. 标签列 ===
    label_columns = ["World Zone"]
    runtime_records = []

    # === 4. 组合参数 ===
    impute_methods = ["knn_auto"]
    reduce_methods = ["UMAP"]

    # === 5. 主循环 ===
    for impute_method in impute_methods:
        for reduce_method in reduce_methods:
            for label_col in label_columns:
                labels = meta[label_col] if label_col in meta.columns else None
                print(f"\n[INFO] Running combination → Impute: {impute_method.upper()} | Reduce: {reduce_method.upper()}")

                start = time.time()
                try:
                    # === 缺失值填补 ===
                    Xi = adna.impute_missing(Xf, method=impute_method)
                    # Xi = adna.grouped_imputation(Xf, labels, method=impute_method)
                    print(f"[OK] Imputation ({impute_method}) complete.")

                    # === 降维 / 懒加载聚类 ===
                    if Xi.empty:
                        print("[INFO] Xi is empty (sharded mode detected) → using lazy clustering + UMAP")
                        latest_dir = max((processed_dir.glob("mode_filled_*")), key=lambda p: p.stat().st_mtime)
                        emb = adna.streaming_umap_from_parquet(latest_dir, n_components=2, max_cols=50000, pca_dim=50)
                        print(f"[OK] Lazy UMAP complete. Shape={emb.shape}")
                        target_matrix = emb

                    else:
                        emb = adna.compute_embeddings(Xi, method=reduce_method, n_components=2, random_state=42)
                        print(f"[OK] {reduce_method.upper()} complete.")
                        target_matrix = Xi

                    # === 绘制降维结果 ===
                    if labels is not None:
                        fig_path = results_dir / f"{impute_method}_embeddings_{reduce_method}_{labels.name}_packed.png"
                        adna.plot_embedding(emb, labels=labels,
                                            title=f"{reduce_method.upper()} ({impute_method}) Projection",
                                            save_path=fig_path)

                    # === 层次聚类分析 ===
                    print("\n[INFO] Running hierarchical clustering analysis...")
                    best_k = adna.find_optimal_clusters(target_matrix, linkage_method="average",
                                                        metric="hamming", cluster_range=range(2, 9))
                    print(f"[INFO] Using optimal cluster number: {best_k}")

                    meta_2d = adna.cluster_on_embedding(emb, meta.copy(), n_clusters=best_k)
                    adna.save_csv(meta_2d, results_dir / f"{impute_method}_{reduce_method}_2D_clusters_packed.csv")

                    adna.plot_cluster_on_embedding(
                        emb, labels=meta_2d["cluster_2D"], meta=meta_2d, label_col="World Zone",
                        title=f"{reduce_method.upper()} + Clustering ({impute_method}) [k={best_k}]"
                    )

                    summary_df = adna.compare_clusters_vs_labels(meta_2d, cluster_col="cluster_2D", label_col="World Zone")
                    adna.save_report(summary_df, results_dir / f"{impute_method}_{reduce_method}_cluster_vs_label_packed.csv")

                    runtime_records.append({
                        "imputation_method": impute_method,
                        "embedding_method": reduce_method,
                        "total_time_s": round(time.time() - start, 2),
                        "status": "success"
                    })

                except Exception as e:
                    traceback.print_exc()
                    print(f"[WARN] Combination failed: {e}")
                    runtime_records.append({
                        "imputation_method": impute_method,
                        "embedding_method": reduce_method,
                        "total_time_s": round(time.time() - start, 2),
                        "status": "failed",
                        "error": str(e)
                    })


    # === 6. 保存运行总结 ===
    if runtime_records:
        adna.save_runtime_report(runtime_records, results_dir / "runtime_summary_packed.csv")

    print("\n[ALL DONE] 所有组合执行完成！")
    print(f"[PATH] 查看结果目录: {results_dir}")


if __name__ == "__main__":
    main()
