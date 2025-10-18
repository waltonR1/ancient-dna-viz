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
- **df**: 投影后的结果，列名为 `Dim1`, `Dim2`等。

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

## 2. genotopython.py – 基因文件读取与转换库

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

### 2.1 loadRawGenoFile

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

### 2.2 unpackfullgenofile

解包完整的 `.geno` 文件，将其转换为 numpy 数组。

**参数：**

|     参数     |  类型   | 是否默认 |      说明      |
|:----------:|:-----:|:----:|:------------:|
| `filename` | `str` |      | `.geno` 文件路径 |

**返回：**

`(geno: np.ndarry, nind: int, nsnp: int, rlen: int)`

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

### 2.3 unpackAndFilterSNPs

解包并筛选指定 SNP 索引的基因型数据。

**参数：**

|      参数      |      类型      | 是否默认 |             说明             |
|:------------:|:------------:|:----:|:--------------------------:|
|    `geno`    | `np.ndarray` |      |      原始 numpy 编码基因型矩阵      |
| `snpIndexes` | `list[int]`  |      | 要保留的 SNP 索引列表（与 .snp 文件对应） |
|    `nind`    |    `int`     |      |            个体数量            |

**返回：**

`geno: np.ndarry`

* **geno**：过滤并解码后的 SNP 数组

**示例：**

```python
import ancient_dna as adna

geno, nind, nsnp, rlen = adna.unpackfullgenofile("data/sample.geno")
filtered = adna.unpackAndFilterSNPs(geno, snpIndexes=[0, 5, 9], nind=nind)
```

---

### 2.4 genofileToCSV

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

### 2.5 genofileToPandas

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

### 2.6 CreateLocalityFile

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

### 2.7 unpack22chrDNAwithLocations

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

`(df: pd.DataFrame | np.ndarry , annowithloc: pd.DataFrame)`

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
### 2.8 unpackYDNAfull

从 `.geno` 文件中提取 Y 染色体 (chromosome 24) 的 SNP 信息。

**参数：**

| 参数             | 类型     | 是否默认    | 说明               |
|----------------|--------|---------|------------------|
| `genofilename` | `str`  |         | `.geno` 文件路径     |
| `snpfilename`  | `str`  |         | `.snp` 文件路径      |
| `indfilename`  | `str`  | `""`    | `.ind` 文件路径（可留空） |
| `transpose`    | `bool` | `True`  | 是否转置输出矩阵         |
| `toCSV`        | `bool` | `False` | 是否导出结果 CSV 文件    |

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
### 2.9 unpackChromosome

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
### 2.10 unpackChromosomefromAnno

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

### 2.11 FilterYhaplIndexes

过滤 Y 染色体样本索引，仅保留符合条件的男性个体。

**参数：**

|        参数        |          类型          |        是否默认         |           说明            |
|:----------------:|:--------------------:|:-------------------:|:-----------------------:|
|     `pdAnno`     |    `pd.DataFrame`    |                     | `.anno` 文件读取的 DataFrame |
| `includefilters` | `list[str]` 或 `None` |       `None`        |   要保留的单倍群关键字（可为 None）   |
| `excludefilters` | `list[str]` 或 `None` | `["na", " ", ".."]` |       要排除的单倍群关键字        |

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
### 2.12 ExtractYHaplogroups

从 `.anno` 文件中提取 Y 染色体单倍群信息。

**参数：**

|        参数        |          类型          |  是否默认  |          说明          |
|:----------------:|:--------------------:|:------:|:--------------------:|
|    `annofile`    |        `str`         |        |     `.anno` 文件路径     |
|   `separator`    |        `str`         | `"\t"` | `.anno` 文件分隔符（默认制表符） |
| `includefilters` | `list[str]` 或 `None` | `None` |      要包含的单倍群关键字      |
| `excludefilters` | `list[str]` 或 `None` | `None` |      要排除的单倍群关键字      |

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
### 2.13 unpackYDNAfromAnno

基于 `.anno` 文件提取 Y 染色体的 SNP 基因型数据。

**参数：**

|        参数        |          类型          |  是否默认   |      说明      |
|:----------------:|:--------------------:|:-------:|:------------:|
|  `genofilename`  |        `str`         |         | `.geno` 文件路径 |
|  `snpfilename`   |        `str`         |         | `.snp` 文件路径  |
|  `annofilename`  |        `str`         |         | `.anno` 文件路径 |
| `includefilters` | `list[str]` 或 `None` | `None`  |  要包含的单倍群关键字  |
| `excludefilters` | `list[str]` 或 `None` | `None`  |  要排除的单倍群关键字  |
|   `transpose`    |        `bool`        | `True`  |   是否转置结果矩阵   |
|     `toCSV`      |        `bool`        | `False` | 是否导出 CSV 文件  |

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
- **ids**: 样本ID序列
- **X**: SNP数值矩阵，行=样本，列=SNP
- **snp_cols**: SNP列名列表

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
- **meta**: 样本注释表

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
- **df**: 读取的 DataFrame

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
- **sample_missing**: 每个样本（行）的缺失率
- **sample_missing**: 每个 SNP（列）的缺失率 

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
- **df**: 过滤后的矩阵

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
- **df**: 填补后的矩阵。

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


### 5.2 build_embedding_report

生成降维嵌入结果的统计报告。

**参数：**

|     参数      |       类型       | 是否默认 |                      说明                      |
|:-----------:|:--------------:|:----:|:--------------------------------------------:|
| `embedding` | `pd.DataFrame` |      | 降维结果 DataFrame，列名通常为 `["Dim1", "Dim2", ...]` |

**返回：**

`pd.DataFrame` 
- 含每维的均值、标准差、最小值、最大值。
- 含字段如下：
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

|    参数     |      类型       | 是否默认 |                                                   说明                                                    |
|:---------:|:-------------:|:----:|:-------------------------------------------------------------------------------------------------------:|
| `records` | `list[dict]`  |      | 每个算法运行时间的记录列表。格式示例：<br>`[{"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52}]` |
|  `path`   | `str \| Path` |      |                                             输出文件路径（包含文件名）。                                              |

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

|          函数名          |       功能简介       |
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

