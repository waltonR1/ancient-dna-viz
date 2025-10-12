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
    执行“单一组合”的数据处理流水线：缺失值填补 → 降维 → 可视化（可选）→ 指标报告 → 运行时记录。

    Args:
        X (np.ndarray): 预处理后的基因型矩阵（对齐且完成缺失率过滤）。
        meta (pd.DataFrame): 与 X 对齐后的元数据表（包含可能用作标签的列）。
        impute_method (str): 缺失值填补方法，例如 'mode'、'mean'、'knn'。
        reduce_method (str): 降维方法，例如 'umap'、'tsne'、'mds'、'isomap'。
        labels (Optional[pd.Series]): 可选的分类标签（如 meta['Y haplogroup']）。为 None 时不绘制着色图。
        processed_dir (Path): 保存填补后数据的目录（会写出 `geno_{impute}.csv`、`meta_{impute}.csv`）。
        results_dir (Path): 保存降维结果、图像与报告的目录（会写出投影 CSV、PNG、report CSV）。

    Returns:
        Dict[str, Any]: 一条运行记录字典，包括：
            {
                "imputation_method": str,
                "embedding_method": str,
                "label": str # 标签列名或 "None"
                "runtime_s": float | None, # 降维耗时（秒）；失败则为 None
                "status": "success" | "failed",
                "error": str | 可无  # 失败时包含错误信息
            }

    Side Effects:
        - 写入以下文件到磁盘：
          * `processed/geno_{impute}.csv`, `processed/meta_{impute}.csv`
          * `results/{impute}_proj_{reduce}.csv`
          * `results/{impute}_proj_{reduce}_{labels.name}.png`（仅当 labels 不为 None）
          * `results/{impute}_embedding_{reduce}_report.csv`
        - 控制台打印进度与耗时。

    Notes:
        - 函数内部捕获降维阶段异常：不会抛出到上层；失败时返回 status='failed' 并包含 error 信息。
        - 绘图仅在 `labels is not None` 时进行；图片文件名会带上标签列名以便区分。
        - 为了可重复性，降维时固定 `random_state=42`；如需修改，可在调用侧调整此函数或封装参数。
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
    sm, cm = adna.compute_missing_rates(X1)
    Xf = adna.filter_by_missing(X1, sm, cm)
    missing_report = adna.build_missing_report(sm, cm)
    adna.save_report(missing_report, results_dir / "missing_report.csv")
    print(f"[OK] Missingness report saved.")

    # === 3. 预留标签列组合（可自由添加/删除） ===
    label_columns = [
        "Y haplogroup",
        "World Zone"
    ]

    # === 4. 需要测试的组合 ===
    impute_methods = ["mode", "mean", "knn"]
    reduce_methods = ["umap", "tsne", "mds", "isomap"]

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