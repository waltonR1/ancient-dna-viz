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
def run_impute_reduce_pipeline(X, meta, impute_method, reduce_method, labels, processed_dir, results_dir):
    """
    运行“缺失值填补 + 降维分析”流程，并保存所有结果文件。

    :param X: 原始基因型矩阵 (pd.DataFrame)，行=样本，列=特征，可能含缺失值。
    :param meta: 样本元数据 (pd.DataFrame)，包含样本的附加信息或标签。
    :param impute_method: 缺失值填补方法名称，如 "mean"、"mode"、"median"。
    :param reduce_method: 降维算法名称，如 "pca"、"tsne"、"umap"。
    :param labels: 分类标签 (pd.Series)，用于绘图着色。若为 None，则不生成图像。
    :param processed_dir: 处理后数据的输出目录 (Path)。
    :param results_dir: 降维结果与报告的输出目录 (Path)。

    :return: 字典对象，包含运行摘要信息：
             {
                 "imputation_method": 填补方法名称,
                 "embedding_method": 降维方法名称,
                 "label": 标签列名（若无则为 "None"）,
                 "runtime_s": 运行耗时（秒）,
                 "status": "success" 或 "failed",
                 "error": 错误信息（仅在失败时存在）
             }

    步骤说明:
        1. 缺失值填补：调用 adna.impute_missing() 对基因矩阵进行缺失值处理。
        2. 保存填补结果：输出至 processed_dir（含 geno_*.csv 与 meta_*.csv）。
        3. 降维计算：调用 adna.compute_embeddings() 执行 PCA / t-SNE / UMAP。
        4. 结果保存：
            - 导出降维坐标 CSV；
            - 若 labels 存在，则绘制并保存散点图；
            - 生成降维结果报告（含方差、坐标统计等）。
        5. 错误捕获：若降维失败，打印警告并返回失败状态字典。

    说明:
        - 每种填补与降维组合都会生成独立结果文件。
        - 文件命名示例：
            geno_mean.csv、mean_proj_umap.csv、mean_embedding_pca_report.csv。
    """
    print(f"\n[STEP] Running combination → Impute: {impute_method.upper()} | Reduce: {reduce_method.upper()}")

    # 1. 缺失值填补
    Xi = adna.impute_missing(X, method=impute_method)
    geno_out = processed_dir / f"geno_{impute_method}.csv"
    meta_out = processed_dir / f"meta_{impute_method}.csv"
    adna.save_csv(Xi, geno_out)
    adna.save_csv(meta, meta_out)
    print(f"[OK] Saved imputed data: {geno_out.name}, {meta_out.name}")

    # 2. 降维
    start = time.time()
    try:
        proj = adna.compute_embeddings(Xi, method=reduce_method, n_components=2, random_state=42)
        elapsed = round(time.time() - start, 3)
        print(f"[OK] {reduce_method.upper()} completed in {elapsed:.2f}s")

        # 保存降维结果
        proj_path = results_dir / f"{impute_method}_proj_{reduce_method}.csv"
        adna.save_csv(proj, proj_path)

        # 绘图（若 labels 存在）
        if labels is not None:
            fig_path = results_dir / f"{impute_method}_proj_{reduce_method}_{labels.name}.png"
            adna.plot_embedding(
                proj,
                labels=labels,
                title=f"{reduce_method.upper()} ({impute_method}) Projection by {labels.name}",
                save_path=fig_path
            )

        # 生成报告
        emb_report = adna.build_embedding_report(proj)
        report_path = results_dir / f"{impute_method}_embedding_{reduce_method}_report.csv"
        adna.save_report(emb_report, report_path)
        print(f"[OK] Embedding report saved: {report_path.name}")

        return {
            "imputation_method": impute_method,
            "embedding_method": reduce_method,
            "label": labels.name if labels is not None else "None",
            "runtime_s": elapsed,
            "status": "success"
        }

    except Exception as e:
        elapsed = round(time.time() - start, 3)
        print(f"[WARN] {reduce_method.upper()} failed on {impute_method}: {e}")
        return {
            "imputation_method": impute_method,
            "embedding_method": reduce_method,
            "label": labels.name if labels is not None else "None",
            "runtime_s": None,
            "error": str(e),
            "status": "failed"
        }


# === 主函数 ===
def main():
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
    print("[STEP] Aligning and computing missingness ...")
    X1, meta1 = adna.align_by_id(ids, X, meta)
    missing_path = results_dir / "Missing.png"
    adna.plot_missing_values(X1,save_path=missing_path)
    print("[OK] Displayed missing value pattern for inspection.")
    sm, cm = adna.compute_missing_rates(X1)
    Xf = adna.filter_by_missing(X1, sm, cm)
    missing_report = adna.build_missing_report(sm, cm)
    adna.save_report(missing_report, results_dir / "missing_report.csv")
    print(f"[OK] Missingness report saved.")
    adna.plot_missing_values(Xf)


    # === 3. 预留标签列组合（可自由添加/删除） ===
    label_columns = [
        "Y haplogroup",
        "World Zone"
    ]

    # === 4. 需要测试的组合 ===
    # impute_methods = ["mode", "mean", "knn"]
    # reduce_methods = ["umap", "tsne", "mds", "isomap"]
    impute_methods = ["mode"]
    reduce_methods = ["umap"]

    runtime_records = []

    # === 5. 自动循环运行所有组合 ===
    for impute_method in impute_methods:
        for reduce_method in reduce_methods:
            for label_col in label_columns:
                labels = meta1[label_col] if label_col in meta1.columns else None
                record = run_impute_reduce_pipeline(
                    X=Xf,
                    meta=meta1,
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
        adna.save_runtime_report(runtime_records, runtime_path)
        print(f"[OK] Runtime summary saved: {runtime_path.name}")

    print("\n[ALL DONE] 所有组合已执行完成！")
    print(f"[PATH] 查看结果目录: {results_dir}")


if __name__ == "__main__":
    main()