# 🧬 Ancient DNA Viz – API Reference

本文件为 `ancient_dna_viz` 项目的开发者接口文档，  
包含四个核心模块：  
`io.py`（数据 I/O）、`preprocess.py`（数据预处理）、  
`visualize.py`（降维与可视化）、`report.py`（结果报告）。

---

## 🧩 1. io.py – 数据输入 / 输出层

| 函数名 | 功能说明 | 输入参数 | 输出结果 |
|:--------|:-----------|:-----------|:-----------|
| **`load_geno(path, id_col="Genetic ID", sep=";")`** | 读取基因型矩阵文件（CSV 格式）。<br>Reads a genotype matrix (samples × SNPs). | `path`: 文件路径；`id_col`: ID 列名；`sep`: 分隔符 | `(ids, X, snp_cols)`：样本ID、矩阵、SNP列名 |
| **`load_meta(path, sep=";")`** | 读取样本注释信息（群体、标签等）。<br>Reads meta data table with sample annotations. | `path`: 文件路径；`sep`: 分隔符 | `pd.DataFrame` |
| **`save_csv(df, path, index=False)`** | 保存 DataFrame 为 CSV 文件。<br>Saves a DataFrame to CSV format. | `df`: 数据表；`path`: 保存路径；`index`: 是否保留索引 | 无返回值（保存文件并打印 `[OK]`） |
| **`load_table(path, sep=";")`** | 通用表格加载函数（任意 CSV / TSV）。<br>Generic table loader. | `path`: 文件路径；`sep`: 分隔符 | `pd.DataFrame` |

---

## 🧮 2. preprocess.py – 数据清洗与缺失处理

| 函数名 | 功能说明 | 输入参数 | 输出结果 |
|:--------|:-----------|:-----------|:-----------|
| **`align_by_id(ids, X, meta, id_col="Genetic ID")`** | 按样本 ID 对齐基因型矩阵与注释表。<br>Aligns genotype matrix and meta info by sample IDs. | `ids`: 样本ID序列；`X`: 矩阵；`meta`: 注释表 | `(X_aligned, meta_aligned)` |
| **`compute_missing_rates(X)`** | 计算样本与 SNP 层面的缺失率。<br>Computes per-sample and per-SNP missing rates. | `X`: 基因型矩阵 | `(sample_missing, snp_missing)` |
| **`filter_by_missing(X, sample_missing, snp_missing, max_sample_missing=0.8, max_snp_missing=0.8)`** | 过滤缺失率超过阈值的样本与位点。<br>Filters out samples/SNPs with too many missing values. | 同上 + 阈值参数 | 过滤后的 DataFrame |
| **`_fill_mode(Z, fallback=1)`** | 按列众数填补缺失值（内部函数）。<br>Fill missing values by column mode. | `Z`: 矩阵；`fallback`: 回退值 | 填补后的矩阵 |
| **`_fill_mean(Z)`** | 按列均值填补缺失值。<br>Fill missing values by column mean. | `Z`: 矩阵 | 填补后的矩阵 |
| **`_fill_knn(Z, n_neighbors=5)`** | 使用 KNNImputer 填补缺失值。<br>KNN-based imputation by sample similarity. | `Z`: 矩阵；`n_neighbors`: 近邻数 | 填补后的矩阵 |
| **`impute_missing(X, method="mode", n_neighbors=5)`** | 缺失值填补主函数，调度 mode/mean/knn 方法。<br>Unified interface for missing-value imputation. | `X`: 矩阵；`method`: 方法类型 | 填补后的矩阵 |

---

## 🎨 3. visualize.py – 降维与可视化

| 函数名 | 功能说明 | 输入参数 | 输出结果 |
|:--------|:-----------|:-----------|:-----------|
| **`_compute_umap(X, n_components=2, **kwargs)`** | 使用 UMAP 计算嵌入。<br>Compute UMAP embeddings. | `X`: 输入矩阵；`n_components`: 降维维度 | `pd.DataFrame` |
| **`_compute_tsne(X, n_components=2, **kwargs)`** | 使用 t-SNE 计算嵌入。<br>Compute t-SNE embeddings. | 同上 | `pd.DataFrame` |
| **`_compute_mds(X, n_components=2, **kwargs)`** | 使用 MDS 计算嵌入。<br>Compute MDS embeddings. | 同上 | `pd.DataFrame` |
| **`_compute_isomap(X, n_components=2, **kwargs)`** | 使用 Isomap 计算嵌入。<br>Compute Isomap embeddings. | 同上 | `pd.DataFrame` |
| **`compute_embeddings(X, method="umap", n_components=2, **kwargs)`** | 降维统一接口。根据 method 自动选择算法。<br>Unified interface for dimensionality reduction. | `X`: 输入矩阵；`method`: 算法名称；`n_components`: 目标维度 | 降维后的嵌入结果 |
| **`plot_embeddings(df, labels=None, title="Embeddings Projection", save_path=None, figsize=(8,6))`** | 绘制降维嵌入结果散点图。<br>Plots the 2D embedding scatter plot. | `df`: 降维结果；`labels`: 标签；`save_path`: 保存路径 | 无返回值（显示或保存图像） |

---

## 📊 4. report.py – 报告生成与导出

| 函数名 | 功能说明 | 输入参数 | 输出结果 |
|:--------|:-----------|:-----------|:-----------|
| **`build_missing_report(sample_missing, snp_missing)`** | 生成缺失率汇总报告。<br>Builds missing rate summary report. | 缺失率 Series | 汇总 DataFrame |
| **`build_embedding_report(embedding)`** | 生成降维嵌入的统计报告（均值、方差、范围）。<br>Builds embedding statistics report. | 降维结果 DataFrame | 统计 DataFrame |
| **`save_report(df, path)`** | 保存报告为 CSV。<br>Saves report table to CSV file. | DataFrame、路径 | 无返回值 |
| **`combine_reports(reports: dict)`** | 合并多份报告（添加模块前缀）。<br>Combines multiple reports into one table. | 报告字典 | 合并后的 DataFrame |

---

## ⚙️ 5. Pipeline 示例

```python
from ancient_dna_viz.io import load_geno, load_meta
from ancient_dna_viz.preprocess import compute_missing_rates
from ancient_dna_viz.report import build_missing_report, build_embedding_report, save_report
from ancient_dna_viz.visualize import compute_embeddings, plot_embeddings

# 1️⃣ 加载数据
ids, X, snp_cols = load_geno("data/Yhaplfiltered40pct.csv")
meta = load_meta("data/Yhaplfiltered40pct_classes.csv")

# 2️⃣ 计算缺失率
sample_missing, snp_missing = compute_missing_rates(X)

# 3️⃣ 生成报告
missing_report = build_missing_report(sample_missing, snp_missing)
save_report(missing_report, "results/reports/missing_report.csv")

# 4️⃣ 降维 + 绘图 + 嵌入报告
emb = compute_embeddings(X, method="umap", n_components=2, random_state=42)
plot_embeddings(emb, labels=meta["haplogroup"], save_path="results/plots/umap.png")
embed_report = build_embedding_report(emb)
save_report(embed_report, "results/reports/embedding_report.csv")
