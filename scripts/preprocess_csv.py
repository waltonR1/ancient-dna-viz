"""
preprocess_csv.py
-----------------
数据预处理与分析主脚本（Pipeline Orchestrator）
Main pipeline script for data preprocessing, imputation, dimensionality reduction, and clustering.

该脚本是 ancient-dna-viz 项目的执行入口，
用于自动化运行完整的 “缺失值填补 → 降维 → 聚类 → 可视化 → 报告生成”流程。
This script orchestrates the full pipeline of the ancient-dna-viz project:
missing-value imputation → dimensionality reduction → clustering → visualization → reporting.

功能 / Functions:
    - run_impute_reduce_pipeline(): 执行单个填补+降维组合流程并保存所有输出结果。
      Run a single imputation + reduction combination and save all generated outputs.
    - main(): 主入口函数，批量运行所有组合、生成报告与聚类分析。
      Main entry function to execute all combinations and generate reports & clustering analyses.

模块角色 / Role in Pipeline:
    该模块位于最上层，整合所有核心功能模块：
        inspect_ancestrymap → 数据完整性验证
        preprocess → 缺失值检测与过滤
        impute → 缺失值填补（多种策略）
        embedding → 降维（UMAP / t-SNE / MDS / Isomap）
        clustering → 层次聚类 + 标签一致性评估
        visualize & report → 图像与结果导出
    This orchestrator ties together all lower-level modules in the pipeline.
"""

import time
import warnings
from pathlib import Path
from scipy.sparse import SparseEfficiencyWarning
import ancient_dna as adna


# === 全局警告屏蔽 ===
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


# === 单一组合执行函数 ===
def run_impute_reduce_pipeline(X, meta, impute_method, reduce_method, labels, processed_dir, results_dir, auto_cluster: bool = True):
    """
    运行“缺失值填补 + 降维分析”完整流程
    Run the full pipeline step (Imputation + Dimensionality Reduction + Clustering).

    :param X: pd.DataFrame
        原始基因型矩阵（行=样本，列=特征，可能包含缺失值）。
        Raw genotype matrix (rows = samples, columns = SNP features).

    :param meta: pd.DataFrame
        样本元数据（包含标签、群体或地理信息）。
        Metadata table containing sample annotations.

    :param impute_method: str
        填补方法名称（如 "mode"、"mean"、"knn_auto"、"knn_faiss" 等）。
        Imputation method name.

    :param reduce_method: str
        降维算法名称（"umap"、"tsne"、"mds"、"isomap"）。
        Dimensionality reduction algorithm name.

    :param labels: pd.Series or None
        样本分类标签，用于散点着色。
        Sample labels used for coloring in plots.

    :param processed_dir: Path
        缺失值填补结果保存目录。
        Directory for imputed data outputs.

    :param results_dir: Path
        降维与聚类结果输出目录。
        Directory for embedding and clustering results.

    :param auto_cluster: bool, default=True
        是否自动确定最佳聚类数。
        Whether to auto-determine optimal number of clusters.

    :return: dict
        包含运行摘要的字典对象，例如：
        {
            "imputation_method": "mode",
            "embedding_method": "umap",
            "label": "World Zone",
            "runtime_s": 6.42,
            "status": "success"
        }

    流程说明 / Workflow:
        1. 缺失值填补 → 调用 adna.impute_missing()
        2. 数据导出 → 保存填补后的 geno/meta CSV
        3. 降维分析 → adna.compute_embeddings() (UMAP/t-SNE/MDS/Isomap)
        4. 聚类分析 → 自动层次聚类 + 可视化 + 一致性分析
        5. 报告导出 → embedding_report + cluster_vs_label 报告
    """
    print(f"\n[INFO] Running combination → Impute: {impute_method.upper()} | Reduce: {reduce_method.upper()}")

    # 1. 缺失值填补
    start_impute = time.time()
    Xi = adna.impute_missing(X, method=impute_method)
    impute_elapsed = round(time.time() - start_impute, 3)
    print(f"[OK] Imputation ({impute_method}) completed in {impute_elapsed:.2f}s")

    geno_out = processed_dir / f"geno_{impute_method}.csv"
    meta_out = processed_dir / f"meta_{impute_method}.csv"
    adna.save_csv(Xi, geno_out)
    adna.save_csv(meta, meta_out)
    print(f"[OK] Saved imputed data: {geno_out.name}, {meta_out.name}")

    # 2. 降维
    start = time.time()
    try:
        embeddings = adna.compute_embeddings(Xi, method=reduce_method, n_components=2, random_state=42)
        elapsed = round(time.time() - start, 3)
        print(f"[OK] {reduce_method.upper()} completed in {elapsed:.2f}s")

        # 保存降维结果
        embeddings_path = results_dir / f"{impute_method}_embeddings_{reduce_method}.csv"
        adna.save_csv(embeddings, embeddings_path)

        # 绘图（若 labels 存在）
        if labels is not None:
            fig_path = results_dir / f"{impute_method}_embeddings_{reduce_method}_{labels.name}.png"
            adna.plot_embedding(
                embeddings,
                labels=labels,
                title=f"{reduce_method.upper()} ({impute_method}) Projection by {labels.name}",
                save_path=fig_path
            )

        # 生成报告
        emb_report = adna.build_embedding_report(embeddings)
        report_path = results_dir / f"{impute_method}_embedding_{reduce_method}_report.csv"
        adna.save_report(emb_report, report_path)



        # 层次聚类分析
        print("\n[INFO] Running hierarchical clustering analysis...")

        if auto_cluster:
            best_k, scores = adna.find_optimal_clusters_embedding(embeddings, linkage_method="average", metric="euclidean", cluster_range=range(2, 10))
            silhouette_plot_path = results_dir / f"{impute_method}_{reduce_method}_silhouette_trend.png"
            adna.plot_silhouette_trend(scores, save_path=silhouette_plot_path)
            print(f"[INFO] Using optimal cluster number: {best_k}")
        else:
            best_k = 10

        # 高维聚类
        # meta_hd = adna.cluster_high_dimensional(Xi, meta.copy(), n_clusters=best_k)
        # adna.save_csv(meta_hd, results_dir / f"{impute_method}_highdim_clusters.csv")

        # 降维结果聚类
        meta_cluster = adna.cluster_on_embedding(embeddings, meta.copy(), n_clusters=best_k)
        adna.save_csv(meta_cluster, results_dir / f"{impute_method}_{reduce_method}_clusters.csv")

        # 聚类可视化
        cluster_fig_path = results_dir / f"{impute_method}_{reduce_method}_clusters.png"
        adna.plot_cluster_on_embedding(
            embeddings,
            labels=meta_cluster["cluster"],
            meta=meta_cluster,
            label_col="World Zone",
            title=f"{reduce_method.upper()} + Hierarchical Clustering ({impute_method}) [k={best_k}]",
            figsize=(8, 6),
            save_path=cluster_fig_path
        )

        # 聚类 vs 标签一致性分析
        enable_cluster_label_analysis = True
        if enable_cluster_label_analysis:
            try:
                summary_df = adna.compare_clusters_vs_labels(
                    meta_cluster, cluster_col="cluster", label_col="World Zone"
                )
                adna.save_report(
                    summary_df,
                    results_dir / f"{impute_method}_{reduce_method}_cluster_vs_label.csv"
                )
                print(f"[OK] Saved cluster-label comparison report.")
            except Exception as e:
                print(f"[WARN] Cluster-label comparison failed: {e}")

        return {
            "imputation_method": impute_method,
            "embedding_method": reduce_method,
            "label": labels.name if labels is not None else "None",
            "impute_time_s": impute_elapsed,
            "reduce_time_s": elapsed,
            "total_time_s": round(impute_elapsed + elapsed, 3),
            "status": "success"
        }

    except Exception as e:
        elapsed = round(time.time() - start, 3)
        print(f"[WARN] {reduce_method.upper()} failed on {impute_method}: {e}")
        return {
            "imputation_method": impute_method,
            "embedding_method": reduce_method,
            "label": labels.name if labels is not None else "None",
            "impute_time_s": impute_elapsed if "impute_elapsed" in locals() else None,
            "reduce_time_s": None,
            "total_time_s": impute_elapsed if "impute_elapsed" in locals() else None,
            "error": str(e),
            "status": "failed"
        }


def main():
    """
    主函数：自动执行完整分析流程
    Main entry function to run the full analysis pipeline.

    任务流程 / Workflow:
        1. 加载基因型矩阵与元数据；
           Load genotype matrix and metadata.
        2. 计算缺失率并进行样本与位点过滤；
           Compute missingness and filter samples/SNPs.
        3. 对各类填补与降维算法组合循环执行；
           Iterate over multiple imputation + embedding combinations.
        4. 自动保存所有 CSV、PNG、报告与聚类结果。
           Automatically save all CSV, PNG, report, and clustering outputs.

    输出 / Outputs:
        - data/processed/*.csv → 缺失值填补后的矩阵
                                 Imputed genotype matrices.
        - data/results/*.csv/.png → 降维结果、聚类分布、质量报告
                                    Embeddings, clustering results, and reports.
        - runtime_summary.csv → 每种算法组合的运行时间汇总
                                Runtime summary for all method combinations.
    """
    ROOT = Path(__file__).resolve().parents[1]
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    results_dir = ROOT / "data" / "results"
    for d in [raw_dir, processed_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === 1. 加载数据 ===
    geno_path = raw_dir / "Yhaplfiltered40pct 1.csv"
    meta_path = raw_dir / "Yhaplfiltered40pct_classes 1.csv"
    print(f"[INFO] Loading data:\n- Geno: {geno_path}\n- Meta: {meta_path}")
    ids, X, _ = adna.load_geno(geno_path)
    meta = adna.load_meta(meta_path)

    # === 2. 对齐与预处理 ===
    X1, meta1 = adna.align_by_id(ids, X, meta)
    missing_path = results_dir / "Missing.png"
    adna.plot_missing_values(X1,save_path=missing_path)

    sm, cm = adna.compute_missing_rates(X1)
    Xf, keep_rows = adna.filter_by_missing(X1, sm, cm)
    metaf = adna.filter_meta_by_rows(meta1, keep_rows)
    missing_report = adna.build_missing_report(sm, cm)
    adna.save_report(missing_report, results_dir / "missing_report.csv")

    after_filtering_missing_path = results_dir / "Missing_after_filtering.png"
    adna.plot_missing_values(Xf,save_path=after_filtering_missing_path)


    # === 3. 预留标签列组合（可自由添加/删除） ===
    # label_columns = [
    #     "Y haplogroup",
    #     "World Zone",
    # ]
    label_columns = [
        "World Zone"
    ]

    # === 4. 需要测试的组合 ===
    impute_methods = ['knn_hamming_balltree']
    reduce_methods = ["tsne"]

    runtime_records = []

    # === 5. 自动循环运行所有组合 ===
    for impute_method in impute_methods:
        for reduce_method in reduce_methods:
            for label_col in label_columns:
                labels = metaf[label_col] if label_col in metaf.columns else None
                record = run_impute_reduce_pipeline(
                    X=Xf,
                    meta=metaf,
                    impute_method=impute_method,
                    reduce_method=reduce_method,
                    labels=labels,
                    processed_dir=processed_dir,
                    results_dir=results_dir
                )
                runtime_records.append(record)

    # === 6. 保存运行时间汇总 ===
    if runtime_records:
        runtime_path = results_dir / "runtime_summary.csv"
        print("\n")
        adna.save_runtime_report(runtime_records, runtime_path)

    print("\n[ALL DONE] All operations have been completed!")
    print(f"[PATH] View Results Directory: {results_dir}")


if __name__ == "__main__":
    main()