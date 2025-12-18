# 🧬 Ancient DNA Visualization Toolkit – API 文档

---
版本：v0.2.0
作者：waltonR1
日期：2025-11-8

本手冊详细介绍了本项目中各模块的函数接口、参数、返回值及使用说明。

---

## 📖 目录（Table of Contents）

- [1. clustering.py – 层次聚类模块](#1-clusteringpy--层次聚类模块)
- [2. embedding.py – 降维算法模块](#2-embeddingpy--降维算法模块)
- [3. genotopython.py – 基因文件读取与转换库](#3-genotopythonpy--基因文件读取与转换库)
- [4. io.py – 数据读写与合并接口](#4-iopy--数据读写与合并接口)
- [5. preprocess.py – 数据预处理与缺失值填补](#5-preprocesspy--数据预处理与缺失值填补)
- [6. summary.py – 数据分析和汇总工具](#6-summarypy--数据分析和汇总工具)
- [7. visualize.py – 可视化绘图工具](#7-visualizepy--可视化绘图工具)
- [附录 A – 常用术语与缩写](#a-常用术语与缩写说明)
- [附录 B – 错误与异常说明](#b-错误与异常说明)
- [附录 C - 文件格式说明](#c-文件格式说明eigenstrat)
- [附录 D - 版本变更记录](#d-版本变更记录)

---


## 1. clustering.py – 层次聚类模块

---

该模块是聚类分析模块，用于在基因型矩阵或降维嵌入空间中执行层次聚类（Hierarchical Clustering），自动确定聚类数、计算聚类纯度。

### 📋 函数总览
|             函数名              |           功能简介            |
|:----------------------------:|:-------------------------:|
|   `find_optimal_clusters`    |         自动搜索最佳聚类数         |
|  `cluster_high_dimensional`  |     在高维 SNP 空间执行层次聚类      |
|    `cluster_on_embedding`    | 在降维结果 (t-SNE / UMAP) 空间聚类 |
| `compare_clusters_vs_labels` |      对比聚类结果与真实标签一致性       |

---

### 1.1 find_optimal_clusters

自动搜索最佳聚类数（基于 **轮廓系数 Silhouette Score**）。
通过遍历不同聚类数 `k`，计算每个聚类方案的平均轮廓系数，从而自动选出最优的聚类数量。

**参数：**

|       参数名        |       类型       |      是否默认      |                              说明                              |
|:----------------:|:--------------:|:--------------:|:------------------------------------------------------------:|
|       `X`        | `pd.DataFrame` |                |                     输入矩阵（行 = 样本，列 = 特征）。                     |
| `linkage_method` |     `str`      |  `"average"`   | 层次聚类的合并策略，如 `"single"`、`"complete"`、`"average"`、`"ward"` 等。  |
|     `metric`     |     `str`      |  `"hamming"`   |                    距离度量方式（适用于二进制或基因型矩阵）。                     |
| `cluster_range`  |    `range`     | `range(2, 11)` |                     搜索的聚类数范围，默认从 2 到 10。                     |

**返回：**

`(best_k, scores)`

* **best_k**：最优聚类数（轮廓系数最高的 k）。
* **scores**：包含所有 `(k, silhouette_score)` 的列表，可用于绘制趋势图。

**算法逻辑：**

1. 遍历给定范围内的聚类数 `k`；
2. 对每个 `k` 执行层次聚类（`AgglomerativeClustering`）；
3. 若聚类结果中包含多个簇，则计算平均轮廓系数；
4. 将 `(k, score)` 记录入列表；
5. 选择轮廓系数最高的聚类数 `best_k` 并返回。

**示例：**

```python
import pandas as pd
from ancient_dna import find_optimal_clusters_embedding

# 示例输入
X = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})

best_k, scores = find_optimal_clusters_embedding(X, linkage_method="average", metric="hamming")

print("最佳聚类数:", best_k)
print("轮廓系数结果:", scores)
```

**说明：**

* 轮廓系数（Silhouette Score）衡量样本与本簇及其他簇的紧密度与分离度；
* 分数越高，聚类效果越好；
* 适用于小至中型数据集的自动聚类优化；
* 可配合 `plot_silhouette_trend()` 一同使用，以可视化评估最优聚类数。

---

### 1.2 cluster_high_dimensional

在**高维 SNP 空间**中执行层次聚类。
直接基于完整填补后的基因型矩阵（未降维）进行聚类分析，用于发现潜在的样本分群结构，并与地理或种群标签进行对比。

**参数：**

|     参数名      |       类型       | 是否默认 |                说明                |
|:------------:|:--------------:|:----:|:--------------------------------:|
| `X_imputed`  | `pd.DataFrame` |      | 已完成缺失值填补的基因型矩阵（行 = 样本，列 = SNP）。  |
|    `meta`    | `pd.DataFrame` |      |    样本元数据表，包含样本信息（如名称、地理区域等）。     |
| `n_clusters` |     `int`      | `5`  |            需要划分的聚类数。             |

**返回：**

`(meta_with_cluster: pd.DataFrame)`

* **meta_with_cluster**：包含聚类结果的元数据表，在原表基础上新增一列 `"cluster"`。

**算法逻辑：**

1. 在已填补缺失值的高维 SNP 矩阵上执行层次聚类；
2. 自动计算轮廓系数（Silhouette Score），用于衡量聚类质量；
3. 将聚类标签结果添加到输入的 `meta` 数据表中；
4. 输出包含 `"cluster"` 列的新元数据表。

**示例：**

```python
import pandas as pd
from ancient_dna import cluster_high_dimensional

# 示例输入
X = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})
meta = pd.DataFrame({
    "SampleID": ["A", "B", "C", "D"],
    "World Zone": ["Europe", "Europe", "Asia", "Asia"]
})

meta_clustered = cluster_high_dimensional(X, meta, n_clusters=3)
print(meta_clustered)
```

**说明：**

* 直接在高维空间聚类，不依赖降维结果；
* 聚类完成后将在控制台输出聚类数与轮廓系数；
* 适用于对比地理区域、单倍群等真实标签的一致性分析；
* 结果可结合 `plot_cluster_on_embedding()` 可视化聚类表现；
* 若数据维度较高，计算量可能较大，建议配合降维后验证结果。

---

### 1.3 cluster_on_embedding

在**降维空间（t-SNE / UMAP 等）**中执行聚类。
基于降维结果的低维嵌入坐标（如二维或三维空间），使用层次聚类方法进行样本分群，用于辅助可视化和聚类一致性分析。

**参数：**

|      参数名       |       类型       | 是否默认 |                   说明                    |
|:--------------:|:--------------:|:----:|:---------------------------------------:|
| `embedding_df` | `pd.DataFrame` |      | 降维后的坐标结果，需包含 `Dim1`, `Dim2`（或 `Dim3`）。  |
|     `meta`     | `pd.DataFrame` |      |            样本元数据表，包含样本的基本信息。            |
|  `n_clusters`  |     `int`      | `5`  |                聚类数（簇数量）。                |


**返回：**

`(meta_with_cluster_2D: pd.DataFrame)`

* **meta_with_cluster_2D**：在原元数据表基础上新增 `"cluster_2D"` 列，记录每个样本在降维空间中的聚类结果。

**算法逻辑：**

1. 基于降维结果 (`embedding_df`) 执行层次聚类；
2. 自动计算聚类的平均轮廓系数（Silhouette Score）；
3. 将聚类标签结果添加到输入的 `meta` 数据表中；
4. 返回包含 `"cluster_2D"` 列的新元数据表。

**示例：**

```python
import pandas as pd
from ancient_dna import cluster_on_embedding

# 示例输入：降维结果 + 样本信息
embedding = pd.DataFrame({
    "Dim1": [0.1, 0.3, 0.8, 1.0],
    "Dim2": [0.2, 0.5, 0.9, 1.2]
})
meta = pd.DataFrame({
    "SampleID": ["A", "B", "C", "D"],
    "World Zone": ["Europe", "Europe", "Asia", "Asia"]
})

meta_clustered = cluster_on_embedding(embedding, meta, n_clusters=2)
print(meta_clustered)
```

**说明：**

* 适用于基于降维结果（UMAP、t-SNE、MDS 等）的样本聚类分析；
* 可用于验证降维可视化与真实标签之间的一致性；
* 聚类结果新增列 `"cluster_2D"`，与 `meta` 表索引顺序保持一致；
* 结果可配合 `plot_cluster_on_embedding()` 进行聚类分布可视化；
* 轮廓系数越高，说明降维空间中的聚类效果越好。

---

### 1.4 compare_clusters_vs_labels

聚类结果与真实标签对比分析。
通过统计每个聚类簇中主标签（**Dominant Label**）及其纯度（**Dominant %**），用于评估聚类结果与真实分类（如地理区域、种群标签等）的对应关系和一致性。

**参数：**

|      参数名      |       类型       |      是否默认      |                    说明                    |
|:-------------:|:--------------:|:--------------:|:----------------------------------------:|
|    `meta`     | `pd.DataFrame` |                |           样本元数据表，需包含聚类列与真实标签列。           |
| `cluster_col` |     `str`      | `"cluster_2D"` | 聚类结果所在列名（如由 `cluster_on_embedding` 生成）。  |
|  `label_col`  |     `str`      | `"World Zone"` |             真实分类标签列名，用于对比分析。             |

**返回：**

`(summary: pd.DataFrame)`

* **summary**：每个聚类簇的组成统计表，包含主标签（Dominant Label）、主标签纯度（Dominant %）及总样本数（Total）。

**算法逻辑：**

1. 对 `cluster_col` 与 `label_col` 进行交叉统计；
2. 计算每个聚类簇中各标签的样本数量；
3. 确定每个簇的主标签（出现次数最多的标签）；
4. 计算主标签的样本占比（纯度 %）；
5. 输出汇总表并打印结果，用于聚类质量评估。

**示例：**

```python
import pandas as pd
from ancient_dna import compare_clusters_vs_labels

# 示例元数据
meta = pd.DataFrame({
    "SampleID": ["A", "B", "C", "D", "E", "F"],
    "World Zone": ["Europe", "Europe", "Asia", "Asia", "Africa", "Africa"],
    "cluster_2D": [0, 0, 1, 1, 2, 2]
})

summary = compare_clusters_vs_labels(meta, cluster_col="cluster_2D", label_col="World Zone")
print(summary)
```

**说明：**

* 纯度（Dominant %）用于衡量聚类结果中主标签的占比；
* 若每个簇的主标签纯度高，说明聚类与真实标签具有良好一致性；
* 适用于验证基因型聚类与地理、种群、生物学标签之间的对应关系；
* 可配合 `plot_cluster_on_embedding()` 一起使用，进行可视化验证；
* 输出的统计表可直接用于报告或后续结果分析。

---

## 2. embedding.py – 降维算法模块

---

该模块提供统一接口与多种降维算法实现（UMAP、t-SNE、MDS、Isomap）。

### 📋 函数总览

|              函数名              |                          功能简介                           |
|:-----------------------------:|:-------------------------------------------------------:|
|     `compute_embeddings`      | 根据指定方法（"umap" / "tsne" / "mds" / "isomap"）执行降维，返回统一格式结果 |
| `streaming_umap_from_parquet` |           通过增量 PCA 与分片 Parquet 文件实现低内存占用的降维流程           |


---

### 2.1 compute_embeddings

统一降维接口，根据 `method` 参数选择算法。

**参数：**

|      参数名       |       类型       | 是否默认 |                      说明                      |
|:--------------:|:--------------:|:----:|:--------------------------------------------:|
|      `X`       | `pd.DataFrame` |      |              基因型矩阵（行=样本，列=SNP）               |
|    `method`    |     `str`      |      | 降维方法：`'umap'`, `'tsne'`, `'mds'`, `'isomap'` |
| `n_components` |     `int`      |      |                 目标维度（2 或 3）                  |
|   `**kwargs`   |       —        |      |                 传递给具体算法的附加参数                 |

**返回：**

`(embedding: pd.DataFrame)` 
- **embedding**: 投影后的结果，列名为 `Dim1`, `Dim2`等。

**示例：**

```python
import pandas as pd
import ancient_dna as adna

X:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })
embedding = adna.compute_embeddings(X, method="umap", n_components=2, random_state=42)
```

---

### 2.2 streaming_umap_from_parquet

低内存版伪流式 UMAP 降维接口，通过 **增量 PCA + 分片 Parquet 文件** 实现对超大基因型矩阵的降维。

**参数：**

|      参数名       |      类型       | 是否默认  |                  说明                   |
|:--------------:|:-------------:|:-----:|:-------------------------------------:|
| `dataset_dir`  | `str \| Path` |       | 分片数据集目录路径（需包含 `columns_index.json`）。  |
| `n_components` |     `int`     |   2   |          降维目标维度（通常为 2 或 3）。           |
|   `max_cols`   |     `int`     | 50000 |        每个分片最多读取的列数，用于控制内存使用量。         |
|   `pca_dim`    |     `int`     |  50   |    先行 PCA 压缩的维数，用于降低后续 UMAP 计算负担。     |
| `random_state` |     `int`     |  42   |            随机种子，用于保证结果可复现。            |

**返回：**

`(embedding: pd.DataFrame)`

* **embedding**：最终的 UMAP 降维结果，列名为 `Dim1`, `Dim2` 等。

**算法流程：**

1. 从 `columns_index.json` 读取分片元数据；
2. 使用 `IncrementalPCA` 逐分片拟合与转换，避免内存峰值；
3. 将所有 PCA 结果拼接为整体降维输入；
4. 在压缩后的矩阵上执行最终的 UMAP 降维；
5. 输出低维投影结果，可用于可视化或聚类分析。

**示例：**

```python
from ancient_dna import streaming_umap_from_parquet

embedding = streaming_umap_from_parquet(
    dataset_dir="data/results/fill_mode/",
    n_components=2,
    max_cols=50000,
    pca_dim=50,
    random_state=42
)

print(embedding.head())
```

---

## 3. genotopython.py – 基因文件读取与转换库

---

该模块提供 `.geno`、`.snp`、`.ind`、`.anno` 等文件的读取、解包、筛选与转换功能。

### 📋 函数总览

|              函数名              |                   功能简介                   |
|:-----------------------------:|:----------------------------------------:|
|       `loadRawGenoFile`       | 读取 `.geno` 文件头信息，提取基本特征信息（样本数、SNP数、每行长度） |
|     `unpackfullgenofile`      |         解包 `.geno` ，转换为 numpy 数组         |
|     `unpackAndFilterSNPs`     |           解包并筛选指定 SNP 索引的基因型数据           |
|        `genofileToCSV`        |          将 `.geno` 文件转换为 CSV 格式          |
|      `genofileToPandas`       |  将 `.geno`、`.snp`、`.ind` 合并为 DataFrame   |
|     `CreateLocalityFile`      |          从 `.anno` 提取个体地理区域与元信息          |
| `unpack22chrDNAwithLocations` |            解包 22 条常染色体并附加地理信息            |
|       `unpackYDNAfull`        |         从`.geno`提取 Y 染色体 SNP 数据          |
|      `unpackChromosome`       |         从`.geno`提取任意指定染色体的SNP数据          |
|  `unpackChromosomefromAnno`   |         从`.anno`文件提取指定染色体的SNP数据          |
|     `FilterYhaplIndexes`      |           从`.anno`过滤 Y 染色体样本索引           |
|     `ExtractYHaplogroups`     |            从`.anno`提取 Y 单倍组列表            |
|     `unpackYDNAfromAnno`      |        基于 `.anno` 文件提取 Y 染色体的SNR         |

---

### 3.1 loadRawGenoFile

读取并准备 `.geno` 文件，提取基本特征信息。

**参数：**

|     参数     |   类型   |  是否默认   |          说明          |
|:----------:|:------:|:-------:|:--------------------:|
| `filename` | `str`  |         | 文件路径，可不带 `.geno` 扩展名 |
|   `ext`    | `bool` | `False` |  是否已包含 `.geno` 扩展名   |

**返回：**

`(geno_file: file, nind: int, nsnp: int, rlen: int)`

* **geno_file**：打开的二进制文件对象
* **nind**：个体数量（样本数）
* **nsnp**：SNP 数量
* **rlen**：每行记录长度（字节数）

**示例：**

```python
import ancient_dna as adna

geno_file, nind, nsnp, rlen = adna.loadRawGenoFile("data/sample")
```

---

### 3.2 unpackfullgenofile

解包完整的 `.geno` 文件，将其转换为 numpy 数组。

**参数：**

|     参数     |  类型   | 是否默认 |      说明      |
|:----------:|:-----:|:----:|:------------:|
| `filename` | `str` |      | `.geno` 文件路径 |

**返回：**

`(geno: np.ndarray, nind: int, nsnp: int, rlen: int)`

* **geno**：解包后的 numpy 数组
* **nind**：个体数量
* **nsnp**：SNP 数量
* **rlen**：每行记录长度

**示例：**

```python
import ancient_dna as adna

geno, nind, nsnp, rlen = adna.unpackfullgenofile("data/sample.geno")
```

---

### 3.3 unpackAndFilterSNPs

解包并筛选指定 SNP 索引的基因型数据。

**参数：**

|      参数      |      类型      | 是否默认 |             说明             |
|:------------:|:------------:|:----:|:--------------------------:|
|    `geno`    | `np.ndarray` |      |      原始 numpy 编码基因型矩阵      |
| `snpIndexes` | `list[int]`  |      | 要保留的 SNP 索引列表（与 .snp 文件对应） |
|    `nind`    |    `int`     |      |            个体数量            |

**返回：**

`geno: np.ndarray`

* **geno**：过滤并解码后的 SNP 数组

**示例：**

```python
import ancient_dna as adna

geno, nind, nsnp, rlen = adna.unpackfullgenofile("data/sample.geno")
filtered = adna.unpackAndFilterSNPs(geno, snpIndexes=[0, 5, 9], nind=nind)
```

---

### 3.4 genofileToCSV

将 `.geno` 文件转换为 CSV 格式。

**参数：**

|     参数     |  类型   | 是否默认  |      说明      |
|:----------:|:-----:|:-----:|:------------:|
| `filename` | `str` |       | `.geno` 文件路径 |
|  `delim`   | `str` | `";"` |   CSV 列分隔符   |

**返回：**

`None`（在原路径下生成 `.csv` 文件）

**示例：**

```python
import ancient_dna as adna

adna.genofileToCSV("data/sample.geno", delim=",")
```

---

### 3.5 genofileToPandas

将 `.geno` 文件转换为 pandas DataFrame。

**参数：**

|      参数       |   类型   |  是否默认  |        说明        |
|:-------------:|:------:|:------:|:----------------:|
|  `filename`   | `str`  |        |   `.geno` 文件路径   |
| `snpfilename` | `str`  |        |   `.snp` 文件路径    |
| `indfilename` | `str`  |        |   `.ind` 文件路径    |
|  `transpose`  | `bool` | `True` | 是否转置矩阵（样本 × SNP） |

**返回：**

`df: pd.DataFrame`

* **df**：转换后的基因型矩阵，索引为样本或 SNP（视转置而定）

**示例：**

```python
import ancient_dna as adna

df = adna.genofileToPandas(
    filename="data/sample.geno",
    snpfilename="data/sample.snp",
    indfilename="data/sample.ind",
    transpose=True
)
```

---

### 3.6 CreateLocalityFile

从 `.anno` 文件中提取个体地理信息并去除重复项。

**参数：**

|       参数       |   类型   |  是否默认   |          说明          |
|:--------------:|:------:|:-------:|:--------------------:|
| `annofilename` | `str`  |         |     `.anno` 文件路径     |
|     `sep`      | `str`  | `"\t"`  |     文件分隔符（默认制表符）     |
|    `toCSV`     | `bool` | `False` |     是否导出为 CSV 文件     |
|   `verbose`    | `bool` | `False` |      是否输出处理进度信息      |
|  `minSNPnbr`   | `int`  |  `-1`   | 最小 SNP 覆盖阈值（过滤低覆盖样本） |
|     `hapl`     | `bool` | `False` |  是否包含 Y/mtDNA 单倍群信息  |

**返回：**

`df: pd.DataFrame`

* **df**：包含地理映射信息的个体表格

**示例：**

```python
import ancient_dna as adna

df = adna.CreateLocalityFile(
    annofilename="data/annotation.anno",
    sep="\t",
    toCSV=True,
    verbose=True,
    minSNPnbr=5000,
    hapl=True
)
```

---

### 3.7 unpack22chrDNAwithLocations

解包前 22 条常染色体 DNA 数据，并整合地理位置信息。
同时支持染色体筛选、单倍群过滤、CSV 导出与内存优化模式。

**参数：**

|        参数        |        类型        |  是否默认   |             说明             |
|:----------------:|:----------------:|:-------:|:--------------------------:|
|  `genofilename`  |      `str`       |         |        `.geno` 文件路径        |
|  `snpfilename`   |      `str`       |         |        `.snp` 文件路径         |
|  `annofilename`  |      `str`       |         |        `.anno` 文件路径        |
|      `chro`      |   `list[int]`    | `None`  |    要提取的染色体编号（默认前 22 条）     |
|   `transpose`    |      `bool`      | `True`  |          是否转置输出矩阵          |
|     `toCSV`      |      `bool`      | `False` |        是否导出 CSV 文件         |
|    `to_numpy`    |      `bool`      | `True`  |    是否返回 numpy 数组（节省内存）     |
|    `verbose`     |      `bool`      | `False` |          是否打印执行进度          |
|   `minSNPnbr`    | `int` \| `float` |  `-1`   | 最小 SNP 覆盖阈值（0<val≤1 表示比例）  |
| `hardhaplfilter` |      `bool`      | `False` | 若含 Y 染色体且为 True，则移除未知单倍群个体 |

**返回：**

`(df: pd.DataFrame | np.ndarray , annowithloc: pd.DataFrame)`

* **df**：DNA 基因型矩阵（类型依 `to_numpy` 而定：`np.ndarray` 或 `pd.DataFrame`）
* **annowithloc**：匹配的地理信息 DataFrame

**说明：**

* 依赖 `CreateLocalityFile()` 获取地区与单倍群信息；
* 若选择包含 Y 染色体，可进行性别与单倍群过滤；
* 内存占用较大，建议一次性导出 CSV 后再使用。

**示例：**

```python
import ancient_dna as adna

df, loc = adna.unpack22chrDNAwithLocations(
    genofilename="data/genotypes.geno",
    snpfilename="data/genotypes.snp",
    annofilename="data/annotation.anno",
    chro=[1, 2, 21],
    transpose=True,
    toCSV=True,
    to_numpy=False,
    verbose=True,
    minSNPnbr=0.8,
    hardhaplfilter=True
)
```

---

### 3.8 unpackYDNAfull

从 `.geno` 文件中提取 Y 染色体 (chromosome 24) 的 SNP 信息。

**参数：**

|        参数        |    类型    |   是否默认    |         说明         |
|:----------------:|:--------:|:---------:|:------------------:|
|  `genofilename`  |  `str`   |           |    `.geno` 文件路径    |
|  `snpfilename`   |  `str`   |           |    `.snp` 文件路径     |
|  `indfilename`   |  `str`   |   `""`    |  `.ind` 文件路径（可留空）  |
|   `transpose`    |  `bool`  |  `True`   |      是否转置输出矩阵      |
|     `toCSV`      |  `bool`  |  `False`  |   是否导出结果 CSV 文件    |

**返回：**

`df: pd.DataFrame`

* **df**：Y 染色体 SNP 基因型矩阵

**说明：**

* 自动识别 `.snp` 文件中 `chromosome = 24` 的行；
* 若提供 `.ind` 文件，则仅保留男性个体；
* 可转置矩阵或导出为 CSV 文件。

**示例：**

```python
import ancient_dna as adna

df_y = adna.unpackYDNAfull(
    genofilename="data/genotypes.geno",
    snpfilename="data/genotypes.snp",
    indfilename="data/genotypes.ind",
    transpose=True,
    toCSV=True
)
```

---

### 3.9 unpackChromosome

从 `.geno` 文件中提取指定染色体 (chrNbr) 的 SNP 数据。

**参数：**

|       参数       |   类型   |  是否默认   |        说明        |
|:--------------:|:------:|:-------:|:----------------:|
| `genofilename` | `str`  |         |   `.geno` 文件路径   |
| `snpfilename`  | `str`  |         |   `.snp` 文件路径    |
|    `chrNbr`    | `int`  |         | 要提取的染色体编号（1–24）  |
| `indfilename`  | `str`  |  `""`   | `.ind` 文件路径（可留空） |
|  `transpose`   | `bool` | `True`  |     是否转置输出矩阵     |
|    `toCSV`     | `bool` | `False` |   是否导出 CSV 文件    |

**返回：**

`df: pd.DataFrame`

* **df**：指定染色体的基因型矩阵

**说明：**

* 自动通过 `.snp` 文件筛选目标染色体 SNP；
* 若提供 `.ind` 文件，将其用于定义样本列；
* 若 `chrNbr=24`，自动调用 `unpackYDNAfull()`；
* 可选择是否转置矩阵或导出为 CSV。

**示例：**

```python
import ancient_dna as adna

df_chr22 = adna.unpackChromosome(
    genofilename="data/genotypes.geno",
    snpfilename="data/genotypes.snp",
    chrNbr=22,
    indfilename="data/genotypes.ind",
    transpose=True,
    toCSV=False
)
```

---

### 3.10 unpackChromosomefromAnno

通过 `.anno` 文件提取指定染色体的 SNP 数据。

**参数：**

|       参数       |   类型   |  是否默认   |      说明      |
|:--------------:|:------:|:-------:|:------------:|
| `genofilename` | `str`  |         | `.geno` 文件路径 |
| `snpfilename`  | `str`  |         | `.snp` 文件路径  |
| `annofilename` | `str`  |         | `.anno` 文件路径 |
|    `chrNbr`    | `int`  |         |   目标染色体编号    |
|  `transpose`   | `bool` | `True`  |   是否转置结果矩阵   |
|    `toCSV`     | `bool` | `False` | 是否导出为 CSV 文件 |

**返回：**

`df: pd.DataFrame`

* **df**：指定染色体的基因型矩阵（行=SNP，列=样本）

**说明：**

* 通过 `.snp` 文件定位目标染色体；
* 依 `.anno` 文件样本信息生成列索引；
* 若染色体为 Y，可使用 `unpackYDNAfromAnno()`；
* 支持转置或导出为 CSV 文件。

**示例：**

```python
import ancient_dna as adna

df_chr1 = adna.unpackChromosomefromAnno(
    genofilename="data/genotypes.geno",
    snpfilename="data/genotypes.snp",
    annofilename="data/annotation.anno",
    chrNbr=1,
    transpose=True,
    toCSV=True
)
```

---

### 3.11 FilterYhaplIndexes

过滤 Y 染色体样本索引，仅保留符合条件的男性个体。

**参数：**

|        参数        |         类型          |        是否默认         |           说明            |
|:----------------:|:-------------------:|:-------------------:|:-----------------------:|
|     `pdAnno`     |   `pd.DataFrame`    |                     | `.anno` 文件读取的 DataFrame |
| `includefilters` | `list[str] \| None` |       `None`        |   要保留的单倍群关键字（可为 None）   |
| `excludefilters` | `list[str] \| None` | `["na", " ", ".."]` |       要排除的单倍群关键字        |

**返回：**

`malesId: list[int]`

* **malesId**：需要保留的男性样本索引列表

**说明：**

* 若指定 `includefilters`，则仅保留匹配该列表的单倍群；
* 默认排除含空格、`na` 或 `..` 等未知单倍群；
* 常作为处理 Y 染色体分析的辅助函数。

**示例：**

```python
import ancient_dna as adna
import pandas as pd

anno = pd.read_csv("data/annotation.anno", sep="\t", low_memory=False)

males = adna.FilterYhaplIndexes(
    pdAnno=anno,
    includefilters=["R1a", "R1b"],
    excludefilters=["na", " ", ".."]
)
```

---

### 3.12 ExtractYHaplogroups

从 `.anno` 文件中提取 Y 染色体单倍群信息。

**参数：**

|        参数        |         类型          |  是否默认  |          说明          |
|:----------------:|:-------------------:|:------:|:--------------------:|
|    `annofile`    |        `str`        |        |     `.anno` 文件路径     |
|   `separator`    |        `str`        | `"\t"` | `.anno` 文件分隔符（默认制表符） |
| `includefilters` | `list[str] \| None` | `None` |      要包含的单倍群关键字      |
| `excludefilters` | `list[str] \| None` | `None` |      要排除的单倍群关键字      |

**返回：**

`(ygroups: pd.Series, malesId: List[int])`

* **ygroups**：符合条件的单倍群序列
* **malesId**：对应的样本索引列表

**说明：**

* 依赖 `FilterYhaplIndexes()` 进行性别与单倍群过滤；
* 可灵活设定包含或排除条件；
* 常用于 Y 染色体分析前的数据准备。

**示例：**

```python
import ancient_dna as adna

ygroups, malesId = adna.ExtractYHaplogroups(
    annofile="data/annotation.anno",
    separator="\t",
    includefilters=["R1a", "R1b"],
    excludefilters=["na", " ", ".."]
)
```

---

### 3.13 unpackYDNAfromAnno

基于 `.anno` 文件提取 Y 染色体的 SNP 基因型数据。

**参数：**

|        参数        |         类型          |  是否默认   |      说明      |
|:----------------:|:-------------------:|:-------:|:------------:|
|  `genofilename`  |        `str`        |         | `.geno` 文件路径 |
|  `snpfilename`   |        `str`        |         | `.snp` 文件路径  |
|  `annofilename`  |        `str`        |         | `.anno` 文件路径 |
| `includefilters` | `list[str] \| None` | `None`  |  要包含的单倍群关键字  |
| `excludefilters` | `list[str] \| None` | `None`  |  要排除的单倍群关键字  |
|   `transpose`    |       `bool`        | `True`  |   是否转置结果矩阵   |
|     `toCSV`      |       `bool`        | `False` | 是否导出 CSV 文件  |

**返回：**

`df: pd.DataFrame`

* **df**：Y 染色体的 SNP 基因型矩阵（行=SNP，列=样本）

**说明：**

* 自动筛选 `.snp` 文件中 `chromosome = 24` 的 SNP；
* 使用 `FilterYhaplIndexes()` 过滤男性样本与指定单倍群；
* 可转置或导出为 CSV 文件。

**示例：**

```python
import ancient_dna as adna

df_y = adna.unpackYDNAfromAnno(
    genofilename="data/genotypes.geno",
    snpfilename="data/genotypes.snp",
    annofilename="data/annotation.anno",
    includefilters=["R1a", "R1b"],
    excludefilters=["na", " ", ".."],
    transpose=True,
    toCSV=True
)
```

---

## 4. io.py – 数据读写与合并接口

---

封装常用的 CSV/表格读取与保存方法。

### 📋 函数总览

|         函数名         |         功能简介          |
|:-------------------:|:---------------------:|
|     `load_geno`     |     读取基因型矩阵（CSV）      |
|     `load_meta`     |        读取样本注释表        |
|     `load_csv`      |  通用 CSV 加载函数（含错误处理）   |
|     `save_csv`      | 导出 DataFrame 为 CSV 文件 |

---

### 4.1 load_geno

读取基因型矩阵。

**参数：**

|    参数    |      类型       |      是否默认      |    说明    |
|:--------:|:-------------:|:--------------:|:--------:|
|  `path`  | `str \| Path` |                |   文件路径   |
| `id_col` |     `str`     | `"Genetic ID"` | 样本 ID 列名 |
|  `sep`   |     `str`     |     `";"`      |   分隔符    |

**返回：**

`(ids: pd.Series, X: pd.DataFrame, snp_cols: List[str])`
- **ids**: 样本ID序列
- **X**: SNP数值矩阵，行=样本，列=SNP
- **snp_cols**: SNP列名列表

**示例：**

```python
import ancient_dna as adna
ids, X, snps = adna.load_geno("data/geno.csv")
```

---

### 4.2 load_meta

读取样本注释表。

**参数：**

|    参数    |      类型       |      是否默认      |    说明    |
|:--------:|:-------------:|:--------------:|:--------:|
|  `path`  | `str \| Path` |                |   文件路径   |
| `id_col` |     `str`     | `"Genetic ID"` | 样本 ID 列名 |
|  `sep`   |     `str`     |     `";"`      |   分隔符    |

**返回：**

`(meta: pd.DataFrame)`
- **meta**: 样本注释表

**示例：**

```python
import ancient_dna as adna
meta = adna.load_meta("data/meta.csv")
```

---

### 4.3 load_csv

通用 CSV 加载函数

**参数：**

|   参数   |      类型       | 是否默认  |  说明  |
|:------:|:-------------:|:-----:|:----:|
| `path` | `str \| Path` |       | 文件路径 |
| `sep`  |     `str`     | `";"` | 分隔符  |

**返回：**

`(df: pd.DataFrame)`
- **df**: 读取的 DataFrame

**示例：**

```python
import ancient_dna as adna
meta = adna.load_csv("data/demo.csv")
```

---

### 4.4 save_csv

导出 DataFrame 为 CSV 文件。

**参数：**

|       参数        |       类型       |  是否默认   |             说明             |
|:---------------:|:--------------:|:-------:|:--------------------------:|
|      `df`       | `pd.DataFrame` |         |      需要导出的 DataFrame       |
|     `path`      | `str \| Path`  |         |            文件路径            |
|      `sep`      |     `str`      |  `";"`  |            分隔符             |
|     `index`     |     `bool`     | `False` |    是否导出 DataFrame 的行索引     |
|    `verbose`    |     `bool`     | `True`  |          是否打印保存信息          |

**返回：**

`None`

**示例：**

```python
import pandas as pd
import ancient_dna as adna

X:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })

adna.save_csv(X, "/geno_out.csv")
```

---

## 5. preprocess.py – 数据预处理与缺失值填补

---

提供数据对齐、缺失率计算与多种填补方法。

### 📋 函数总览

|           函数名           |      功能简介       |
|:-----------------------:|:---------------:|
|      `align_by_id`      | 对齐样本 ID，保留共有样本  |
| `compute_missing_rates` | 计算样本与 SNP 的缺失率  |
|   `filter_by_missing`   | 按阈值过滤高缺失率样本/SNP |
|    `impute_missing`     |    缺失值填补统一接口    |
|  `grouped_imputation`   |     按标签分组填补     |

---

### 5.1 align_by_id

对齐样本 ID，保留共有样本

**参数：**

|    参数    |       类型       |      是否默认      |      说明       |
|:--------:|:--------------:|:--------------:|:-------------:|
|  `ids`   |  `pd.Series`   |                |   样本 ID 序列    |
|   `X`    | `pd.DataFrame` |                |     基因型矩阵     |
|  `meta`  | `pd.DataFrame` |                |      注释表      |
| `id_col` |     `str`      | `"Genetic ID"` | 注释表中的样本 ID 列名 |

**返回：**

`(X_aligned: pd.DataFrame, meta_aligned: pd.DataFrame)`
- **X_aligned**: 仅保留共有样本后的基因型矩阵
- **meta_aligned**: 与 X_aligned 行顺序一致的注释表

**示例：**

```python
import pandas as pd
import ancient_dna as adna

ids: pd.Series = pd.Series(["A", "B", "A", "D"])
X:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })
meta:pd.DataFrame = pd.DataFrame({
        "Genetic ID": ["A", "B", "A", "D"],
        "Y haplogroup": [2, 321, 12312, 421]
    })
X1, meta1 = adna.align_by_id(ids, X, meta)
```

---

### 5.2 compute_missing_rates

计算缺失率（样本维度 & SNP 维度）。
- 0 = 参考等位基因
- 1 = 变异等位基因
- 3 = 缺失

**参数：**

|    参数    |       类型       |      是否默认      |      说明       |
|:--------:|:--------------:|:--------------:|:-------------:|
|   `X`    | `pd.DataFrame` |                |     基因型矩阵     |


**返回：**

`(sample_missing: pd.Series, snp_missing: pd.Series)`
- **sample_missing**: 每个样本（行）的缺失率
- **snp_missing**: 每个 SNP（列）的缺失率 

**示例：**

```python
import pandas as pd
import ancient_dna as adna

X1:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })
sm, cm = adna.compute_missing_rates(X1)
```

---

### 5.3 filter_by_missing

按缺失率阈值过滤样本与SNP

**参数：**

|          参数          |       类型       | 是否默认  |      说明      |
|:--------------------:|:--------------:|:-----:|:------------:|
|         `X`          | `pd.DataFrame` |       |    基因型矩阵     |
|   `sample_missing`   |  `pd.Series`   |       |   每个样本的缺失率   |
|    `snp_missing`     |  `pd.Series`   |       | 每个 SNP 的缺失率  |
| `max_sample_missing` |    `float`     | `0.8` |  样本级最大缺失率阈值  |
|  `max_snp_missing`   |    `float`     | `0.8` | SNP 级最大缺失率阈值 |

**返回：**

`(X_filtered: pd.DataFrame)`
- **X_filtered**: 过滤后的矩阵

**示例：**

```python
import pandas as pd
import ancient_dna as adna

X1:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })
sm: pd.Series = pd.Series([0.7, 0.2, 0.4, 0.1])
cm: pd.Series = pd.Series([0.55, 0.85, 0.16, 0.17])
X_filtered = adna.filter_by_missing(X1, sm, cm)
```

---

### 5.4 impute_missing

执行缺失值填补。

**参数：**

|      参数       |       类型       |   是否默认   |                                                          说明                                                           |
|:-------------:|:--------------:|:--------:|:---------------------------------------------------------------------------------------------------------------------:|
|      `X`      | `pd.DataFrame` |          |                                                         基因型矩阵                                                         |
|   `method`    |     `str`      | `"mode"` | 填补方法（`mode`, `mean`, `knn`, `knn_hamming`,`knn_hamming_abs`,`knn_hamming_adaptive`,`knn_hybrid_autoalpha`,`knn_auto`） |
| `n_neighbors` |     `int`      |   `5`    |                                                      KNN 插补的近邻数                                                       |

**返回：**
`(filled: pd.DataFrame)` 
- **filled**: 填补后的矩阵。

**示例：**

```python
import pandas as pd
import ancient_dna as adna

X:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })
filled = adna.impute_missing(X, method="knn")
```

---

### 5.5 grouped_imputation

按外部标签分组执行缺失值填补（封装版）。
根据给定的外部分组标签（如地理区域、单倍群等），将样本划分为若干子集，并在每个分组内独立执行缺失值填补。
若未提供标签，则执行全局填补。

**参数：**

|   参数名    |         类型          |   是否默认   |                        说明                        |
|:--------:|:-------------------:|:--------:|:------------------------------------------------:|
|   `X`    |   `pd.DataFrame`    |          |             原始基因型矩阵（行 = 样本，列 = SNP）。             |
| `labels` | `pd.Series \| None` |          |      外部分组标签列，如世界区域或单倍群分类。若为 `None`，则执行全局填补。      |
| `method` |        `str`        | `"mode"` | 缺失值填补方法（如 `"mode"`、`"knn_hamming_adaptive"` 等）。  |

**返回：**

`(filled_X: pd.DataFrame)`

* **filled_X**：分组填补后的完整矩阵，索引顺序与原矩阵一致。

**算法逻辑：**

1. 若 `labels=None`，直接对整个矩阵执行全局填补；
2. 否则根据 `labels` 的取值将样本划分为多个子集；
3. 对每个分组单独调用 `impute_missing()` 执行缺失值填补；
4. 对样本数量较小的分组（≤5），自动改用列众数填补；
5. 对样本过少且方法为 `"knn_faiss"` 的情况，降级为 `"mode"` 填补；
6. 最后将所有分组结果重新合并，并按原始索引排序输出。

**示例：**

```python
import pandas as pd
import ancient_dna as adna

X = pd.DataFrame({
    "SNP1": [0, 1, None, 3],
    "SNP2": [3, None, 1, 0],
    "SNP3": [1, 3, 3, None]
})
labels = pd.Series(["Europe", "Europe", "Asia", "Asia"], name="Region")

filled = adna.grouped_imputation(X, labels=labels, method="mode")
print(filled)
```

---

## 6. summary.py – 数据分析和汇总工具

---

该模块用于自动生成数据分析报告、降维结果统计及运行时间汇总表。

### 📋 函数总览

|           函数名            |      功能简介      |
|:------------------------:|:--------------:|
|  `build_missing_report`  |   生成缺失率统计汇总表   |
| `build_embedding_report` | 计算降维结果的数值分布统计  |
|      `save_report`       | 保存报告表格为 CSV 文件 |
|  `save_runtime_report`   |  保存算法运行时间记录表   |

---

### 6.1 build_missing_report

生成样本与 SNP 缺失率汇总表。

**参数：**

|        参数        |     类型      | 是否默认 |     说明      |
|:----------------:|:-----------:|:----:|:-----------:|
| `sample_missing` | `pd.Series` |      |  每个样本的缺失率   |
|  `snp_missing`   | `pd.Series` |      | 每个 SNP 的缺失率 |

**返回：**

`df: pd.DataFrame` 
- 含均值、中位数、最大值的单行统计报告。
- 含单行汇总数据的表格，字段如下：
  - **sample_count**: 样本总数
  - **snp_count**: SNP 总数
  - **sample_missing_mean**: 样本缺失率均值
  - **sample_missing_median**: 样本缺失率中位数
  - **sample_missing_max**: 样本缺失率最大值
  - **snp_missing_mean**: 位点缺失率均值
  - **snp_missing_median**: 位点缺失率中位数
  - **snp_missing_max**: 位点缺失率最大值

**示例：**

```python
import pandas as pd
import ancient_dna as adna

sample_missing: pd.Series = pd.Series([0.7, 0.2, 0.4, 0.1])
snp_missing: pd.Series = pd.Series([0.7, 0.2, 0.4, 0.1])
report = adna.build_missing_report(sample_missing, snp_missing)
```

---

### 6.2 build_embedding_report

生成降维嵌入结果的统计报告。

**参数：**

|     参数      |       类型       | 是否默认 |                      说明                      |
|:-----------:|:--------------:|:----:|:--------------------------------------------:|
| `embedding` | `pd.DataFrame` |      | 降维结果 DataFrame，列名通常为 `["Dim1", "Dim2", ...]` |

**返回：**

`pd.DataFrame` 
- 包含每维的均值、标准差、最小值、最大值四项，字段如下：
  - **Dimension**: 维度名称
  - **Mean**: 平均值
  - **StdDev**: 标准差
  - **Min**: 最小值
  - **Max**: 最大值

**示例：**

```python
import pandas as pd
import ancient_dna as adna

embedding: pd.DataFrame = pd.DataFrame({
    "Dim1": [0.1, 0.2, 0.3, 0.4],
    "Dim2": [-0.5, -0.3, 0.0, 0.2]
})
report = adna.build_embedding_report(embedding)
```

---

### 6.3 save_report

保存报告表格为 CSV 文件。

**参数：**

|       参数        |       类型       |  是否默认   |             说明             |
|:---------------:|:--------------:|:-------:|:--------------------------:|
|      `df`       | `pd.DataFrame` |         |      需要导出的 DataFrame       |
|     `path`      | `str \| Path`  |         |            文件路径            |

**返回：**

`None`

**示例：**

```python
import pandas as pd
import ancient_dna as adna

report: pd.DataFrame = pd.DataFrame({
    "sample_count": [100],
    "snp_count": [50000],
    "sample_missing_mean": [0.12],
    "snp_missing_mean": [0.08]
})

adna.save_report(report, "data/results/missing_report.csv")
```

---

### 6.4 save_runtime_report

保存降维与填补方法运行时间统计表。

**参数：**

|    参数     |      类型       | 是否默认 |                                                 说明                                                  |
|:---------:|:-------------:|:----:|:---------------------------------------------------------------------------------------------------:|
| `records` | `list[dict]`  |      | 每个算法运行时间的记录列表。格式示例：`[{"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52}]` |
|  `path`   | `str \| Path` |      |                                           输出文件路径（包含文件名）。                                            |

**返回：**

`None`

**示例：**

```python
import ancient_dna as adna

records = [
    {"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52},
    {"imputation_method": "mean", "embedding_method": "pca", "runtime_s": 1.84}
]
adna.save_runtime_report(records, "data/results/runtime_summary.csv")

```

---

## 7. visualize.py – 可视化绘图工具

---

该模块用于绘制降维散点图、缺失数据分布等分析图形。

### 📋 函数总览

|             函数名             |        功能简介         |
|:---------------------------:|:-------------------:|
|      `plot_embedding`       |  绘制降维结果散点图（支持 2D）   |
|    `plot_missing_values`    |     可视化缺失值分布矩阵      |
| `plot_cluster_on_embedding` | 绘制聚类结果叠加图，并显示主标签与纯度 |
|   `plot_silhouette_trend`   | 绘制聚类数与平均轮廓系数的关系趋势图  |

---

### 7.1 plot_embedding

绘制二维降维散点图，支持自定义图例位置与颜色映射。
超出 legend_max 的类别在图中与 legend 中均以灰色表示。

**参数：**

|       参数       |          类型           |          是否默认          |                   说明                    |
|:--------------:|:---------------------:|:----------------------:|:---------------------------------------:|
|      `df`      |    `pd.DataFrame`     |                        |            含 Dim1、Dim2 的降维结果            |
|    `labels`    |      `pd.Series`      |                        |                 分类标签，可选                 |
|    `title`     |         `str`         |                        |                   图标题                   |
|  `save_path`   | `str \| Path \| None` |                        |              保存路径（为空则直接显示）              |
|   `figsize`    |        `tuple`        |       `(10, 7)`        |                  图像大小                   |
|  `legend_pos`  |         `str`         |       `"right"`        | 图例位置：`right`, `bottom`, `top`, `inside` |
|     `cmap`     |         `str`         |       `"tab20"`        |                  颜色映射表                  |
|  `legend_max`  |         `int`         |          `20`          |                 最大显示类别数                 |
| `legend_sort`  |        `bool`         |         `True`         |                是否按样本数量排序                |
| `others_color` |        `tuple`        | `(0.7, 0.7, 0.7, 0.5)` |            超出legend限制的样本的颜色             |


**返回：**

`None`

**示例：**

```python
import pandas as pd
import ancient_dna as adna

embedding: pd.DataFrame = pd.DataFrame({
    "Dim1": [0.1, 0.2, 0.3, 0.4],
    "Dim2": [-0.5, -0.3, 0.0, 0.2]
})
meta: pd.Series = pd.Series(["A", "B", "A", "D"])
adna.plot_embedding(embedding, labels=meta, title="UMAP Projection")
```

---

### 7.2 plot_missing_values

智能绘制缺失数据可视化图，根据矩阵规模自动切换绘制模式：小矩阵绘制逐像素缺失图，大矩阵绘制缺失率分布直方图。

**参数：**

|       参数        |          类型           |    是否默认     |          说明           |
|:---------------:|:---------------------:|:-----------:|:---------------------:|
|      `df`       |    `pd.DataFrame`     |             |        基因样本数据         |
|   `save_path`   | `str \| Path \| None` |             |     保存路径（为空则直接显示）     |
| `missing_value` |         `int`         |     `3`     |         缺失值标记         |
|    `figsize`    |        `tuple`        |  `(10, 7)`  |         图像大小          |
| `cmap_present`  |         `str`         | `"#d95f02"` |        非缺失值颜色         |
| `cmap_missing`  |         `str`         | `"#ffffff"` |         缺失值颜色         |
|  `show_ratio`   |        `bool`         |   `True`    |     是否同时显示缺失比例条形图     |
|  `max_pixels`   |         `int`         |    `5e7`    | 当矩阵元素数超过该阈值时自动使用聚合模式。 |



**返回：**

`None`


**示例：**

```python
import pandas as pd
import ancient_dna as adna

X:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })

adna.plot_missing_values(X, "results/missing_values.png")
```

---

### 7.3 plot_cluster_on_embedding

聚类结果叠加可视化。
在降维结果的嵌入空间中绘制聚类分布图，并在每个簇中心标注**主标签（Dominant Label）**及其**纯度（Dominant %）**，用于直观展示聚类质量与标签一致性。

**参数：**

|      参数名       |           类型           |              是否默认               |             说明             |
|:--------------:|:----------------------:|:-------------------------------:|:--------------------------:|
| `embedding_df` |     `pd.DataFrame`     |                                 | 降维结果，需包含 `Dim1`、`Dim2` 列。  |
|    `labels`    |      `pd.Series`       |                                 |     聚类标签，每个样本对应一个簇编号。      |
|     `meta`     | `pd.DataFrame \| None` |             `None`              |   样本元数据（可选），用于计算主标签和纯度。    |
|  `label_col`   |         `str`          |         `"World Zone"`          |   元数据中真实标签列名，用于评估聚类一致性。    |
|    `title`     |         `str`          | `"Clusters on Embedding Space"` |            图标题。            |
|   `figsize`    |        `tuple`         |            `(8, 6)`             |           图像大小。            |
|  `save_path`   |     `Path \| None`     |             `None`              |     若提供路径则保存图片，否则直接显示。     |

**返回：**

无（绘制或保存聚类可视化图）。

**算法逻辑：**

1. 检查输入数据中是否包含 `Dim1` 与 `Dim2`；
2. 按聚类标签绘制散点图，每种簇使用不同颜色；
3. 若提供 `meta` 数据：

   * 计算每个簇的中心坐标；
   * 统计每簇中各真实标签的样本数；
   * 确定主标签及其占比（纯度 %）；
   * 在簇中心处绘制主标签与纯度标注；
4. 输出出版级聚类可视化图，可保存或直接显示。

**示例：**

```python
import pandas as pd
from pathlib import Path
from ancient_dna import plot_cluster_on_embedding

# 模拟降维与聚类结果
embedding = pd.DataFrame({
    "Dim1": [0.1, 0.3, 0.8, 1.0],
    "Dim2": [0.2, 0.5, 0.9, 1.2]
})
labels = pd.Series([0, 0, 1, 1])
meta = pd.DataFrame({
    "World Zone": ["Europe", "Europe", "Asia", "Asia"]
})

# 绘制结果并保存
plot_cluster_on_embedding(
    embedding_df=embedding,
    labels=labels,
    meta=meta,
    label_col="World Zone",
    title="示例：聚类结果叠加图",
    save_path=Path("results/cluster_plot.png")
)
```

**说明：**

* 点颜色代表不同聚类簇；
* 若提供元数据，可计算每个簇中主标签的占比（即纯度）；
* 小簇或标签混杂的区域会显示较低纯度；
* 输出结果可用于评估降维聚类的质量与标签一致性；
* 若不指定保存路径，将直接在屏幕中显示图像。

---

### 7.4 plot_silhouette_trend

绘制**轮廓系数（Silhouette Score）**随聚类数变化的趋势图。
用于帮助选择最优聚类数（k），通过可视化不同聚类数量下的轮廓得分，评估聚类质量与稳定性。

**参数：**

|     参数名     |            类型             |  是否默认  |                      说明                       |
|:-----------:|:-------------------------:|:------:|:---------------------------------------------:|
|  `scores`   | `list[tuple[int, float]]` |        | 聚类数与对应轮廓系数的列表，每个元素为 `(k, silhouette_score)`。  |
| `save_path` |      `Path \| None`       | `None` |              若指定路径则保存图片，否则直接显示。               |

**返回：**

无（绘制或保存趋势图）。

**算法逻辑：**

1. 从输入的 `(k, score)` 列表中提取聚类数与对应轮廓系数；
2. 绘制折线图，横轴为聚类数 `k`，纵轴为轮廓系数；
3. 自动设置网格、标题与坐标标签；
4. 若提供 `save_path`，保存图像至指定路径；
5. 否则直接在屏幕上显示结果。

**示例：**

```python
from pathlib import Path
from ancient_dna import plot_silhouette_trend

# 模拟不同聚类数对应的轮廓系数
scores = [(2, 0.41), (3, 0.52), (4, 0.49), (5, 0.45), (6, 0.43)]

# 绘制并保存结果
plot_silhouette_trend(scores, save_path=Path("results/silhouette_trend.png"))
```

**说明：**

* 轮廓系数（Silhouette Score）越高，表示聚类结构越清晰、类间差异越大；
* 通常选取**得分最高的 k 值**作为最佳聚类数；
* 该图适用于 KMeans、Agglomerative 等聚类模型的结果评估；

---

## 📚 附录

---

### A. 常用术语与缩写说明

|        术语/缩写         |                                   说明                                   |
|:--------------------:|:----------------------------------------------------------------------:|
|         SNP          |                单核苷酸多态性（Single Nucleotide Polymorphism）                 |
|   `.geno`, `.snp`等   |                         EIGENSTRAT 格式的基因型数据文件                          |
|    降维 (Embedding)    |                        将高维基因数据映射到 2D/3D 空间以便可视化                        |
|    缺失值 (Missing)     |                     数据中缺乏或无法识别的等位基因，本项目中统一用 `3` 表示                     |
|   单倍群 (Haplogroup)   |                         表示 Y 染色体或线粒体 DNA 的进化支系                         |

---

### B. 错误与异常说明

|          异常类型          |                     可能触发原因                      |                     解决建议                     |
|:----------------------:|:-----------------------------------------------:|:--------------------------------------------:|
|  `FileNotFoundError`   |            文件路径错误或缺失（如 `load_csv()`）            |              确认路径拼写是否正确，文件是否存在               |
|      `ValueError`      |               传入不合法的参数值，如降维方法拼写错误               |              检查方法名是否符合文档中指定的选项               |
|       `KeyError`       |              DataFrame 中访问了不存在的列名               |        检查 ID 列名是否正确，如 `"Genetic ID"`         |
|     `RuntimeError`     |              运行时出错，如数据文件格式错误或读取失败               |            确认输入数据是否标准化，尝试逐步 debug            |
|      `TypeError`       |          参数类型不符，如传入了非 `DataFrame` 对象等           |         检查参数是否为正确类型，如 `pd.DataFrame`         |

---


### C. 文件格式说明（EIGENSTRAT）

|     文件类型     |     后缀     |                   说明                   |
|:------------:|:----------:|:--------------------------------------:|
|     基因矩阵     |  `.geno`   |        编码为 0, 1, 3，分别代表等位基因/缺失         |
|   SNP 位点信息   |   `.snp`   |          每行对应一个 SNP 的位置与染色体信息          |
|     个体信息     |   `.ind`   |              每个个体的性别与族群信息              |
|     注释信息     |  `.anno`   |            样本的地理、年代、单倍群等元信息            |

### D. 版本变更记录

|   版本   |     日期     |                       说明                       |
|:------:|:----------:|:----------------------------------------------:|
| v0.1.0 | 2025-10-16 | 第一版 API 文档，包含 embedding / genotopython 等核心模块。  |
| v0.1.1 | 2025-10-24 |        修正文档中的返回值命名错误、示例调用函数名错误、若干拼写问题。         |
| v0.2.0 | 2025-11-08 |            clustering模块和针对大数据集函数的添加            |