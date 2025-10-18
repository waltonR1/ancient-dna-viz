# 🧬 Ancient DNA Visualization Toolkit – API 文档

版本：v1.0
作者：waltonR1
日期：2025-10-16

本手冊详细介绍了本项目中各模块的函数接口、参数、返回值及使用说明。
文档结构遵循以下顺序：

1. embedding.py – 降维算法模块
2. genotopython.py – 基因文件读取与转换库
3. io.py – 数据读写与合并接口
4. preprocess.py – 数据预处理与缺失值填补
5. report.py – 报告生成与汇总工具
6. visualize.py – 可视化绘图工具

---

## 1. embedding.py – 降维算法模块

该模块提供统一接口与多种降维算法实现（UMAP、t-SNE、MDS、Isomap）。

### 📋 函数总览

|          函数名           |                           功能简介                            |
|:----------------------:|:---------------------------------------------------------:|
|  `compute_embeddings`  |  根据指定方法（"umap" / "tsne" / "mds" / "isomap"）执行降维，返回统一格式结果  |

---

### 1.1 compute_embeddings

统一降维接口，根据 `method` 参数选择算法。

**参数：**

|      参数名       |       类型       | 是否默认 |                      说明                      |
|:--------------:|:--------------:|:----:|:--------------------------------------------:|
|      `X`       | `pd.DataFrame` |      |              基因型矩阵（行=样本，列=SNP）               |
|    `method`    |     `str`      |      | 降维方法：`'umap'`, `'tsne'`, `'mds'`, `'isomap'` |
| `n_components` |     `int`      |      |                 目标维度（2 或 3）                  |
|   `**kwargs`   |       —        |      |                 传递给具体算法的附加参数                 |

**返回：**

`(df: pd.DataFrame)` 
- 投影后的结果，列名为 `Dim1`, `Dim2`等。

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

## 2️⃣ genotopython.py – 基因文件读取与转换库

该模块提供 `.geno`、`.snp`、`.ind`、`.anno` 等文件的读取、解包、筛选与转换功能。

### 📋 函数总览

| 函数名                           | 功能简介                                  |
|:-----------------------------:|:-------------------------------------:|
| `loadRawGenoFile`             | 读取 `.geno` 文件头信息（样本数、SNP数等）           |
| `unpackfullgenofile`          | 解包 `.geno` 文件为 numpy 数组               |
| `unpackAndFilterSNPs`         | 解包并筛选指定 SNP 索引                        |
| `genofileToCSV`               | 将 `.geno` 文件转换为 CSV 格式                |
| `genofileToPandas`            | 将 `.geno`、`.snp`、`.ind` 合并为 DataFrame |
| `CreateLocalityFile`          | 从 `.anno` 提取个体地理区域与元信息                |
| `unpack22chrDNAwithLocations` | 解包 22 条常染色体并附加地理信息                    |
| `unpackYDNAfull`              | 提取 Y 染色体 SNP 数据                       |
| `unpackChromosome`            | 提取任意指定染色体                             |
| `FilterYhaplIndexes`          | 过滤男性个体与 haplogroup                    |
| `ExtractYHaplogroups`         | 提取 Y 单倍组列表                            |
| `unpackYDNAfromAnno`          | 基于 `.anno` 文件提取 Y 染色体数据               |

---

### 示例：`genofileToPandas(filename, snpfilename, indfilename, transpose=True)`

将 `.geno` 文件转为 Pandas DataFrame。

**参数：**

|      参数名      |   类型   |       说明        |
|:-------------:|:------:|:---------------:|
|  `filename`   | `str`  |  `.geno` 文件路径   |
| `snpfilename` | `str`  |   `.snp` 文件路径   |
| `indfilename` | `str`  |   `.ind` 文件路径   |
|  `transpose`  | `bool` | 是否转置矩阵（默认 True） |

**返回：**
`pd.DataFrame` – 样本×SNP 矩阵。

**示例：**

```python
import ancient_dna as adna
df = adna.genofileToPandas("data/sample", "data/sample.snp", "data/sample.ind")
```

---

## 3. io.py – 数据读写与合并接口

封装常用的 CSV/表格读取与保存方法。

### 📋 函数总览

|         函数名         |         功能简介          |
|:-------------------:|:---------------------:|
|     `load_geno`     |     读取基因型矩阵（CSV）      |
|     `load_meta`     |        读取样本注释表        |
|     `load_csv`      |  通用 CSV 加载函数（含错误处理）   |
|     `save_csv`      | 导出 DataFrame 为 CSV 文件 |

---

### 3.1 load_geno

读取基因型矩阵。

**参数：**

|    参数    |      类型       |      是否默认      |    说明    |
|:--------:|:-------------:|:--------------:|:--------:|
|  `path`  | `str \| Path` |                |   文件路径   |
| `id_col` |     `str`     | `"Genetic ID"` | 样本 ID 列名 |
|  `sep`   |     `str`     |     `";"`      |   分隔符    |

**返回：**

`(ids: pd.Series, X: pd.DataFrame, snp_cols: List[str])`
- ids: 样本ID序列
- X: SNP数值矩阵，行=样本，列=SNP
- snp_cols: SNP列名列表

**示例：**

```python
import ancient_dna as adna
ids, X, snps = adna.load_geno("data/geno.csv")
```

### 3.2 load_meta

读取样本注释表。

**参数：**

|    参数    |      类型       |      是否默认      |    说明    |
|:--------:|:-------------:|:--------------:|:--------:|
|  `path`  | `str \| Path` |                |   文件路径   |
| `id_col` |     `str`     | `"Genetic ID"` | 样本 ID 列名 |
|  `sep`   |     `str`     |     `";"`      |   分隔符    |

**返回：**

`(meta: pd.DataFrame)`
- 样本注释表

**示例：**

```python
import ancient_dna as adna
meta = adna.load_meta("data/meta.csv")
```

### 3.3 load_csv

通用 CSV 加载函数

**参数：**

|   参数   |      类型       | 是否默认  |  说明  |
|:------:|:-------------:|:-----:|:----:|
| `path` | `str \| Path` |       | 文件路径 |
| `sep`  |     `str`     | `";"` | 分隔符  |

**返回：**

`(df: pd.DataFrame)`
- 读取的 DataFrame

**示例：**

```python
import ancient_dna as adna
meta = adna.load_csv("data/demo.csv")
```

### 3.4 save_csv

读取样本注释表。

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

## 4. preprocess.py – 数据预处理与缺失值填补

提供数据对齐、缺失率计算与多种填补方法。

### 📋 函数总览

|           函数名           |      功能简介       |
|:-----------------------:|:---------------:|
|      `align_by_id`      | 对齐样本 ID，保留共有样本  |
| `compute_missing_rates` | 计算样本与 SNP 的缺失率  |
|   `filter_by_missing`   | 按阈值过滤高缺失率样本/SNP |
|    `impute_missing`     |    缺失值填补统一接口    |

---

### 4.1 align_by_id

对齐样本 ID，保留共有样本

**参数：**

|    参数    |       类型       |      是否默认      |      说明       |
|:--------:|:--------------:|:--------------:|:-------------:|
|  `ids`   |  `pd.Series`   |                |   样本 ID 序列    |
|   `X`    | `pd.DataFrame` |                |     基因型矩阵     |
|  `meta`  | `pd.DataFrame` |     `";"`      |      注释表      |
| `id_col` |     `str`      | `"Genetic ID"` | 注释表中的样本 ID 列名 |

**返回：**

`(X_aligned: pd.DataFrame, meta_aligned: pd:DataFrame)`
- X_aligned: 仅保留共有样本后的基因型矩阵
- meta_aligned: 与 X_aligned 行顺序一致的注释表

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

### 4.2 compute_missing_rates

计算缺失率（样本维度 & SNP 维度）。
- 0 = 参考等位基因
- 1 = 变异等位基因
- 3 = 缺失

**参数：**

|    参数    |       类型       |      是否默认      |      说明       |
|:--------:|:--------------:|:--------------:|:-------------:|
|   `X`    | `pd.DataFrame` |                |     基因型矩阵     |


**返回：**

`(sample_missing: pd.Series, sample_missing: pd.Series)`
- sample_missing: 每个样本（行）的缺失率
- sample_missing: 每个 SNP（列）的缺失率 

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

### 4.3 filter_by_missing

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

`(df: pd.DataFrame)`
- 过滤后的矩阵

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
Xf = adna.filter_by_missing(X1, sm, cm)
```


### 4.4 impute_missing

执行缺失值填补。

**参数：**

|      参数       |       类型       |   是否默认   |             说明              |
|:-------------:|:--------------:|:--------:|:---------------------------:|
|      `X`      | `pd.DataFrame` |          |            基因型矩阵            |
|   `method`    |     `str`      | `"mode"` | 填补方法（`mode`, `mean`, `knn`） |
| `n_neighbors` |     `int`      |   `5`    |         KNN 插补的近邻数          |

**返回：**
`(df: pd.DataFrame)` 
- 填补后的矩阵。

**示例：**

```python
import pandas as pd
import ancient_dna as adna

X:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })
Xi = adna.impute_missing(X, method="knn")
```


## 5. summary.py – 数据分析和汇总工具

该模块用于自动生成数据分析报告、降维结果统计及运行时间汇总表。

### 📋 函数总览

|           函数名            |      功能简介      |
|:------------------------:|:--------------:|
|  `build_missing_report`  |   生成缺失率统计汇总表   |
| `build_embedding_report` | 计算降维结果的数值分布统计  |
|      `save_report`       | 保存报告表格为 CSV 文件 |
|  `save_runtime_report`   |  保存算法运行时间记录表   |

---

### 5.1 build_missing_report

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
  - sample_count: 样本总数
  - snp_count: SNP 总数
  - sample_missing_mean: 样本缺失率均值
  - sample_missing_median: 样本缺失率中位数
  - sample_missing_max: 样本缺失率最大值
  - snp_missing_mean: 位点缺失率均值
  - snp_missing_median: 位点缺失率中位数
  - snp_missing_max: 位点缺失率最大值

**示例：**

```python
import pandas as pd
import ancient_dna as adna

sample_missing: pd.Series = pd.Series([0.7, 0.2, 0.4, 0.1])
snp_missing: pd.Series = pd.Series([0.7, 0.2, 0.4, 0.1])
report = adna.build_missing_report(sample_missing, snp_missing)
```


### 5.2 build_embedding_report

生成降维嵌入结果的统计报告。

**参数：**

|     参数      |       类型       | 是否默认 |              说明               |
|:-----------:|:--------------:|:----:|:-----------------------------:|
| `embedding` | `pd.DataFrame` |      | 降维结果 DataFrame，列名通常为 `["Dim1", "Dim2", ...]` |

**返回：**

`pd.DataFrame` 
- 含每维的均值、标准差、最小值、最大值。
- 含字段如下：
  - Dimension: 维度名称
  - Mean: 平均值
  - StdDev: 标准差
  - Min: 最小值
  - Max: 最大值

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

### 5.3 save_report

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

X:pd.DataFrame = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, 1],
        "SNP3": [3, 3, 1, 0]
    })

adna.save_csv(X, "/report.csv")
```


### 5.4 save_runtime_report

保存降维与填补方法运行时间统计表。

**参数：**

|     参数    |       类型      | 是否默认 |                                                    说明                                                   |
| :-------: | :-----------: | :--: | :-----------------------------------------------------------------------------------------------------: |
| `records` |  `list[dict]` |      | 每个算法运行时间的记录列表。格式示例：<br>`[{"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52}]` |
|   `path`  | `str \| Path` |      |                                              输出文件路径（包含文件名）。                                             |

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

## 6. visualize.py – 可视化绘图工具

该模块用于绘制降维散点图、缺失数据分布等分析图形。

### 📋 函数总览

|          函数名          | 功能简介             |
|:---------------------:|:----------------:|
|   `plot_embedding`    | 绘制降维结果散点图（支持 2D） |
| `plot_missing_values` |    可视化缺失值分布矩阵    |

---

### 6.1 plot_embedding

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

### 6.2 plot_missing_values

绘制缺失数据矩阵图，白色代表缺失值，可同时显示缺失比例条形图。

**参数：**

|       参数        |          类型           |    是否默认     |      说明       |
|:---------------:|:---------------------:|:-----------:|:-------------:|
|      `df`       |    `pd.DataFrame`     |             |    基因样本数据     |
|   `save_path`   | `str \| Path \| None` |             | 保存路径（为空则直接显示） |
| `missing_value` |         `int`         |     `3`     |     缺失值标记     |
|    `figsize`    |        `tuple`        |  `(10, 7)`  |     图像大小      |
| `cmap_present`  |         `str`         | `"#d95f02"` |    非缺失值颜色     |
| `cmap_missing`  |         `str`         | `"#ffffff"` |     缺失值颜色     |
|  `show_ratio`   |        `bool`         |   `True`    | 是否同时显示缺失比例条形图 |


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

adna.plot_missing_values(X, "/report.csv")
```

---

## 📚 附录

* 文档生成日期：2025-10-16
* 基于函数内中文 docstring 自动汇编
* 适用于项目：**ancient-dna-viz**

