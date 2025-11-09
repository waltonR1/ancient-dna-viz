# 🧬 Ancient DNA Visualization Toolkit – API Reference

---
Version: v0.2.0
Author: waltonR1
Date: 2025-11-8

This manual provides detailed documentation of all modules, function interfaces, parameters, return values, and usage instructions.

---

## 📖 Table of Contents

* [1. clustering.py – Hierarchical Clustering Module](#1-clusteringpy--hierarchical-clustering-module)
* [2. embedding.py – Dimensionality Reduction Algorithms](#2-embeddingpy--dimensionality-reduction-algorithms)
* [3. genotopython.py – Genotype File Reading and Conversion Library](#3-genotopythonpy--genotype-file-reading-and-conversion-library)
* [4. io.py – Data Reading and Merging Interface](#4-iopy--data-reading-and-merging-interfaces)
* [5. preprocess.py – Data Preprocessing and Missing-Value Imputation](#5-preprocesspy--data-preprocessing-and-missing-value-imputation)
* [6. summary.py – Data Analysis and Summary Tools](#6-summarypy--data-analysis-and-summary-tools)
* [7. visualize.py – Visualization Tools](#7-visualizepy--visualization-tools)
* [Appendix A – Common Terms and Abbreviations](#a-common-terms-and-abbreviations)
* [Appendix B – Errors and Exceptions](#b-error-and-exception-reference)
* [Appendix C – File Format Specification](#c-file-format-reference-eigenstrat)
* [Appendix D – Version Change Log](#d-version-history)

---

## 1. clustering.py – Hierarchical Clustering Module

---

This module provides clustering analysis functions for executing hierarchical clustering (Hierarchical Clustering) on genotype matrices or embedding spaces, automatically determining the optimal number of clusters and computing cluster purity.

### 📋 Function Overview

|        Function Name         |                               Description                                |
|:----------------------------:|:------------------------------------------------------------------------:|
|   `find_optimal_clusters`    |         Automatically search for the optimal number of clusters          |
|  `cluster_high_dimensional`  |      Perform hierarchical clustering in high-dimensional SNP space       |
|    `cluster_on_embedding`    | Perform clustering in reduced-dimensional (t-SNE / UMAP) embedding space |
| `compare_clusters_vs_labels` |           Compare cluster results with true label consistency            |

---

### 1.1 find_optimal_clusters

Automatically search for the optimal number of clusters (based on **Silhouette Score**).
By iterating over different numbers of clusters `k`, it calculates the average silhouette score for each clustering scheme and automatically selects the optimal number of clusters.

**Parameters:**

|    Parameter     |      Type      |    Default     |                                Description                                 |
|:----------------:|:--------------:|:--------------:|:--------------------------------------------------------------------------:|
|       `X`        | `pd.DataFrame` |                |             Input matrix (rows = samples, columns = features).             |
| `linkage_method` |     `str`      |  `"average"`   | Linkage strategy such as `"single"`, `"complete"`, `"average"`, `"ward"`.  |
|     `metric`     |     `str`      |  `"hamming"`   |        Distance metric (suitable for binary or genotype matrices).         |
| `cluster_range`  |    `range`     | `range(2, 11)` |         Range of cluster numbers to search (default from 2 to 10).         |

**Returns:**

`(best_k, scores)`

* **best_k**: Optimal number of clusters (the k with the highest silhouette score).
* **scores**: A list containing all `(k, silhouette_score)` pairs, which can be used to plot trend charts.

**Algorithm Logic:**

1. Iterate over each clustering number `k`;
2. Perform hierarchical clustering (`AgglomerativeClustering`) for each `k`;
3. If the clustering result contains multiple clusters, compute the average silhouette score;
4. Record `(k, score)` into a list;
5. Select the clustering number `best_k` with the highest silhouette score and return it.

**Example:**

```python
import pandas as pd
from ancient_dna import find_optimal_clusters

# Example input
X = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})

best_k, scores = find_optimal_clusters(X, linkage_method="average", metric="hamming")

print("Optimal cluster number:", best_k)
print("Silhouette score results:", scores)
```

**Notes:**

* The silhouette score measures compactness within clusters and separation between clusters.
* A higher score indicates better clustering performance.
* Suitable for small to medium-sized datasets for automatic clustering optimization.
* Can be used with `plot_silhouette_trend()` to visualize and evaluate the optimal cluster number.

---

### 1.2 cluster_high_dimensional

Perform hierarchical clustering in **high-dimensional SNP space**.
Directly conducts clustering analysis based on the fully imputed genotype matrix (without dimensionality reduction) to discover potential population structures and compare with geographic or population labels.

**Parameters:**

|  Parameter   |      Type      | Default |                                    Description                                     |
|:------------:|:--------------:|:-------:|:----------------------------------------------------------------------------------:|
| `X_imputed`  | `pd.DataFrame` |         |             Imputed genotype matrix (rows = samples, columns = SNPs).              |
|    `meta`    | `pd.DataFrame` |         | Sample metadata table containing sample information (e.g., names, regions, etc.).  |
| `n_clusters` |     `int`      |   `5`   |                         Number of clusters to divide into.                         |

**Returns:**

`(meta_with_cluster: pd.DataFrame)`

* **meta_with_cluster**: The metadata table containing clustering results, with an additional column `"cluster"`.

**Algorithm Logic:**

1. Perform hierarchical clustering on the imputed high-dimensional SNP matrix;
2. Automatically compute the silhouette score to evaluate clustering quality;
3. Add clustering label results to the input `meta` table;
4. Output a new metadata table containing the `"cluster"` column.

**Example:**

```python
import pandas as pd
from ancient_dna import cluster_high_dimensional

# Example input
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

**Notes:**

* Performs clustering directly in high-dimensional space without dimensionality reduction.
* Outputs the number of clusters and silhouette score in the console.
* Useful for comparing with geographic regions, haplogroups, or true labels.
* Results can be visualized using `plot_cluster_on_embedding()` to display cluster distribution.
* For very high-dimensional data, computation may be heavy; consider validating results after dimensionality reduction.

---

### 1.3 cluster_on_embedding

Perform clustering in **low-dimensional embedding space (t-SNE / UMAP, etc.)**.
Based on low-dimensional embedding coordinates (e.g., 2D or 3D), hierarchical clustering is applied to group samples, assisting visualization and consistency analysis.

**Parameters:**

|   Parameter    |      Type      | Default |                       Description                        |
|:--------------:|:--------------:|:-------:|:--------------------------------------------------------:|
| `embedding_df` | `pd.DataFrame` |         | Embedding result containing `Dim1`, `Dim2` (or `Dim3`).  |
|     `meta`     | `pd.DataFrame` |         |   Sample metadata table with basic sample information.   |
|  `n_clusters`  |     `int`      |   `5`   |                   Number of clusters.                    |

**Returns:**

`(meta_with_cluster_2D: pd.DataFrame)`

* **meta_with_cluster_2D**: Metadata table with an added `"cluster_2D"` column recording cluster results in the embedding space.

**Algorithm Logic:**

1. Perform hierarchical clustering on the embedding results (`embedding_df`);
2. Automatically compute the average silhouette score;
3. Add clustering label results to the input `meta` table;
4. Return the new metadata table containing the `"cluster_2D"` column.

**Example:**

```python
import pandas as pd
from ancient_dna import cluster_on_embedding

# Example input: embedding results + sample information
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

**Notes:**

* Suitable for clustering analysis based on dimensionality reduction results (UMAP, t-SNE, MDS, etc.).
* Can be used to verify consistency between visualization and true labels.
* The clustering result adds a `"cluster_2D"` column, preserving the same order as the input `meta` table.
* The results can be visualized using `plot_cluster_on_embedding()` for cluster distribution.
* A higher silhouette score indicates better clustering in the embedding space.

---

### 1.4 compare_clusters_vs_labels

Analyze the relationship between clustering results and true labels.
By counting the dominant label (**Dominant Label**) and its purity (**Dominant %**) in each cluster, it evaluates how well the clustering results align with true classifications (e.g., geographic regions or populations).

**Parameters:**

|   Parameter   |      Type      |    Default     |                                         Description                                         |
|:-------------:|:--------------:|:--------------:|:-------------------------------------------------------------------------------------------:|
|    `meta`     | `pd.DataFrame` |                |                 Sample metadata table containing cluster and label columns.                 |
| `cluster_col` |     `str`      | `"cluster_2D"` | Name of the column containing cluster results (e.g., generated by `cluster_on_embedding`).  |
|  `label_col`  |     `str`      | `"World Zone"` |                  Column name of true classification labels for comparison.                  |

**Returns:**

`(summary: pd.DataFrame)`

* **summary**: Cluster composition statistics for each cluster, including dominant label, dominant purity percentage, and total sample count.

**Algorithm Logic:**

1. Perform cross-tabulation between `cluster_col` and `label_col`;
2. Count the number of samples for each label within each cluster;
3. Determine the dominant label (the label with the highest count) for each cluster;
4. Calculate the proportion of the dominant label (purity %);
5. Output a summary table and print results for cluster quality evaluation.

**Example:**

```python
import pandas as pd
from ancient_dna import compare_clusters_vs_labels

# Example metadata
meta = pd.DataFrame({
    "SampleID": ["A", "B", "C", "D", "E", "F"],
    "World Zone": ["Europe", "Europe", "Asia", "Asia", "Africa", "Africa"],
    "cluster_2D": [0, 0, 1, 1, 2, 2]
})

summary = compare_clusters_vs_labels(meta, cluster_col="cluster_2D", label_col="World Zone")
print(summary)
```

**Notes:**

* Purity (**Dominant %**) measures the proportion of the dominant label in each cluster.
* High purity indicates good alignment between clustering results and true labels.
* Useful for validating genotype clustering against geographic, population, or biological labels.
* Can be combined with `plot_cluster_on_embedding()` for visual validation.
* The output summary table can be directly used for reports or further analysis.

---

## 2. embedding.py – Dimensionality Reduction Algorithms

---

This module provides a unified interface and multiple dimensionality reduction algorithms (UMAP, t-SNE, MDS, Isomap).

### 📋 Function Overview

|         Function Name         |                                                             Description                                                             |
|:-----------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
|     `compute_embeddings`      | Execute dimensionality reduction using specified method ("umap" / "tsne" / "mds" / "isomap") and return results in a unified format |
| `streaming_umap_from_parquet` |                      Perform low-memory dimensionality reduction using incremental PCA and Parquet file shards                      |

---

### 2.1 compute_embeddings

Unified dimensionality reduction interface that selects the algorithm based on the `method` parameter.

**Parameters:**

|   Parameter    |      Type      | Default |                                Description                                |
|:--------------:|:--------------:|:-------:|:-------------------------------------------------------------------------:|
|      `X`       | `pd.DataFrame` |         |             Genotype matrix (rows = samples, columns = SNPs).             |
|    `method`    |     `str`      |         | Dimensionality reduction method: `'umap'`, `'tsne'`, `'mds'`, `'isomap'`. |
| `n_components` |     `int`      |         |                      Target dimensionality (2 or 3).                      |
|   `**kwargs`   |       —        |         |          Additional parameters passed to the specific algorithm.          |

**Returns:**

`(embedding: pd.DataFrame)`

* **embedding**: The projection result with columns such as `Dim1`, `Dim2`, etc.

**Example:**

```python
import pandas as pd
import ancient_dna as adna

X = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})
embedding = adna.compute_embeddings(X, method="umap", n_components=2, random_state=42)
```

---

### 2.2 streaming_umap_from_parquet

Low-memory pseudo-streaming UMAP interface, achieving dimensionality reduction on large genotype matrices using **incremental PCA + Parquet file shards**.

**Parameters:**

|   Parameter    |      Type       | Default |                              Description                              |                                                                       
|:--------------:|:---------------:|:-------:|:---------------------------------------------------------------------:|
| `dataset_dir`  | `str  \|  Path` |         | Directory path of dataset shards (must include `columns_index.json`). |  
| `n_components` |      `int`      |    2    |                Target dimensionality (usually 2 or 3).                |                                                                       
|   `max_cols`   |      `int`      |  50000  |     Maximum number of columns per shard to control memory usage.      |                                                                       
|   `pca_dim`    |      `int`      |   50    | Number of PCA dimensions retained before UMAP to reduce computation.  |                                                                       
| `random_state` |      `int`      |   42    |                   Random seed for reproducibility.                    |                                                                       


**Returns:**

`(embedding: pd.DataFrame)`

* **embedding**: Final UMAP result with columns such as `Dim1`, `Dim2`, etc.

**Algorithm Flow:**

1. Read shard metadata from `columns_index.json`;
2. Use `IncrementalPCA` to fit and transform each shard incrementally, avoiding memory peaks;
3. Concatenate all PCA results as input for dimensionality reduction;
4. Perform final UMAP reduction on the compressed matrix;
5. Output low-dimensional projection results for visualization or clustering analysis.

**Example:**

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

## 3. genotopython.py – Genotype File Reading and Conversion Library

---

This module provides functions for reading, unpacking, filtering, and converting files such as `.geno`, `.snp`, `.ind`, and `.anno`.

### 📋 Function Overview

|         Function Name         |                                        Description                                         |
|:-----------------------------:|:------------------------------------------------------------------------------------------:|
|       `loadRawGenoFile`       | Read `.geno` file header and extract basic metadata (sample count, SNP count, row length). |
|     `unpackfullgenofile`      |                      Unpack `.geno` file and convert to numpy array.                       |
|     `unpackAndFilterSNPs`     |                 Unpack and filter genotype data by specified SNP indexes.                  |
|        `genofileToCSV`        |                            Convert `.geno` file to CSV format.                             |
|      `genofileToPandas`       |                 Merge `.geno`, `.snp`, and `.ind` into a pandas DataFrame.                 |
|     `CreateLocalityFile`      |            Extract individual geographic and metadata information from `.anno`.            |
| `unpack22chrDNAwithLocations` |             Unpack 22 autosomal chromosomes and attach geographic information.             |
|       `unpackYDNAfull`        |                        Extract Y chromosome SNP data from `.geno`.                         |
|      `unpackChromosome`       |                Extract SNP data for any specified chromosome from `.geno`.                 |
|  `unpackChromosomefromAnno`   |                     Extract chromosome-specific SNP data from `.anno`.                     |
|     `FilterYhaplIndexes`      |                      Filter Y chromosome sample indexes from `.anno`.                      |
|     `ExtractYHaplogroups`     |                          Extract Y haplogroup list from `.anno`.                           |
|     `unpackYDNAfromAnno`      |                      Extract Y chromosome SNRs based on `.anno` file.                      |

---

### 3.1 loadRawGenoFile

Read and prepare `.geno` file, extracting basic characteristics.

**Parameters:**

| Parameter  |  Type  | Default |                  Description                   |
|:----------:|:------:|:-------:|:----------------------------------------------:|
| `filename` | `str`  |         |     File path, may omit `.geno` extension.     |
|   `ext`    | `bool` | `False` | Whether `.geno` extension is already included. |

**Returns:**

`(geno_file: file, nind: int, nsnp: int, rlen: int)`

* **geno_file**: Opened binary file object.
* **nind**: Number of individuals (samples).
* **nsnp**: Number of SNPs.
* **rlen**: Row length (in bytes).

**Example:**

```python
import ancient_dna as adna

geno_file, nind, nsnp, rlen = adna.loadRawGenoFile("data/sample")
```

---

### 3.2 unpackfullgenofile

Unpack a complete `.geno` file and convert it to a numpy array.

**Parameters:**

| Parameter  | Type  | Default |      Description      |
|:----------:|:-----:|:-------:|:---------------------:|
| `filename` | `str` |         | Path to `.geno` file. |

**Returns:**

`(geno: np.ndarray, nind: int, nsnp: int, rlen: int)`

* **geno**: Unpacked numpy array.
* **nind**: Number of individuals.
* **nsnp**: Number of SNPs.
* **rlen**: Record length per line.

**Example:**

```python
import ancient_dna as adna

geno, nind, nsnp, rlen = adna.unpackfullgenofile("data/sample.geno")
```

---

### 3.3 unpackAndFilterSNPs

Unpack and filter genotype data by specified SNP indexes.

**Parameters:**

|  Parameter   |     Type     | Default |                          Description                          |
|:------------:|:------------:|:-------:|:-------------------------------------------------------------:|
|    `geno`    | `np.ndarray` |         |            Original numpy-encoded genotype matrix.            |
| `snpIndexes` | `list[int]`  |         | List of SNP indexes to retain (corresponding to `.snp` file). |
|    `nind`    |    `int`     |         |                    Number of individuals.                     |

**Returns:**

`geno: np.ndarray`

* **geno**: Filtered and decoded SNP array.

**Example:**

```python
import ancient_dna as adna

geno, nind, nsnp, rlen = adna.unpackfullgenofile("data/sample.geno")
filtered = adna.unpackAndFilterSNPs(geno, snpIndexes=[0, 5, 9], nind=nind)
```

---

### 3.4 genofileToCSV

Convert `.geno` file to CSV format.

**Parameters:**

| Parameter  | Type  | Default |      Description      |
|:----------:|:-----:|:-------:|:---------------------:|
| `filename` | `str` |         | Path to `.geno` file. |
|  `delim`   | `str` |  `";"`  | CSV column separator. |

**Returns:**

`None` (Generates `.csv` file in the original path.)

**Example:**

```python
import ancient_dna as adna

adna.genofileToCSV("data/sample.geno", delim=",")
```

---

### 3.5 genofileToPandas

Convert `.geno` file into a pandas DataFrame.

**Parameters:**

|   Parameter   |       Type       | Default |                    Description                    |
|:-------------:|:----------------:|:-------:|:-------------------------------------------------:|
|  `filename`   |      `str`       |         |               Path to `.geno` file.               |
| `snpfilename` |      `str`       |         |               Path to `.snp` file.                |
| `indfilename` |      `str`       |         |               Path to `.ind` file.                |
|  `transpose`  | `bool` \| `True` |         | Whether to transpose the matrix (samples × SNPs). |

**Returns:**

`df: pd.DataFrame`

* **df**: Converted genotype matrix, indexed by sample or SNP depending on transpose option.

**Example:**

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

Extract individual geographic information from `.anno` file and remove duplicates.

**Parameters:**

|   Parameter    |  Type  | Default |                          Description                          |
|:--------------:|:------:|:-------:|:-------------------------------------------------------------:|
| `annofilename` | `str`  |         |                     Path to `.anno` file.                     |
|     `sep`      | `str`  | `"\t"`  |                 File delimiter (default tab).                 |
|    `toCSV`     | `bool` | `False` |                Whether to export as CSV file.                 |
|   `verbose`    | `bool` | `False` |                Whether to print progress info.                |
|  `minSNPnbr`   | `int`  |  `-1`   | Minimum SNP coverage threshold (filter low-coverage samples). |
|     `hapl`     | `bool` | `False` |      Whether to include Y/mtDNA haplogroup information.       |

**Returns:**

`df: pd.DataFrame`

* **df**: Individual table containing geographic mapping information.

**Example:**

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

Unpack DNA data for the first 22 autosomal chromosomes and integrate geographic location information.
Supports chromosome selection, haplogroup filtering, CSV export, and memory optimization mode.

**Parameters:**

|    Parameter     |     Type      | Default |                                    Description                                     |                                                            
|:----------------:|:-------------:|:-------:|:----------------------------------------------------------------------------------:|
|  `genofilename`  |     `str`     |         |                               Path to `.geno` file.                                |                                                            
|  `snpfilename`   |     `str`     |         |                                Path to `.snp` file.                                |                                                            
|  `annofilename`  |     `str`     |         |                               Path to `.anno` file.                                |                                                            
|      `chro`      |  `list[int]`  | `None`  |                 Chromosome numbers to extract (default first 22).                  |                                                            
|   `transpose`    |    `bool`     | `True`  |                        Whether to transpose output matrix.                         |                                                            
|     `toCSV`      |    `bool`     | `False` |                            Whether to export CSV file.                             |                                                            
|    `to_numpy`    |    `bool`     | `True`  |                    Whether to return numpy array (save memory).                    |                                                            
|    `verbose`     |    `bool`     | `False` |                             Whether to print progress.                             |                                                            
|   `minSNPnbr`    | `int\| float` |  `-1`   |             Minimum SNP coverage threshold (0<val≤1 means proportion).             |
| `hardhaplfilter` |    `bool`     | `False` | If Y chromosome is included and True, remove individuals with unknown haplogroups. |                                                           

**Returns:**

`(df: pd.DataFrame | np.ndarray , annowithloc: pd.DataFrame)`

* **df**: DNA genotype matrix (type depends on `to_numpy`: `np.ndarray` or `pd.DataFrame`).
* **annowithloc**: Matching geographic information DataFrame.

**Notes:**

* Depends on `CreateLocalityFile()` for region and haplogroup information.
* If Y chromosome is included, allows filtering by sex or haplogroup.
* High memory usage; recommended to export to CSV once for reuse.

**Example:**

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

Extract Y chromosome (chromosome 24) SNP information from a `.geno` file.

**Parameters:**

|   Parameter    |  Type  | Default |                 Description                 |
|:--------------:|:------:|:-------:|:-------------------------------------------:|
| `genofilename` | `str`  |         |            Path to `.geno` file.            |
| `snpfilename`  | `str`  |         |            Path to `.snp` file.             |
| `indfilename`  | `str`  |  `""`   |       Path to `.ind` file (optional).       |
|  `transpose`   | `bool` | `True`  |   Whether to transpose the output matrix.   |
|    `toCSV`     | `bool` | `False` | Whether to export the result to a CSV file. |

**Returns:**

`df: pd.DataFrame`

* **df**: Y chromosome SNP genotype matrix.

**Notes:**

* Automatically identifies rows in `.snp` where `chromosome = 24`.
* If an `.ind` file is provided, only male individuals are retained.
* Supports transposing the matrix or exporting it to CSV.

**Example:**

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

Extract SNP data of a specified chromosome (chrNbr) from a `.geno` file.

**Parameters:**

|   Parameter    |  Type  | Default |               Description               |
|:--------------:|:------:|:-------:|:---------------------------------------:|
| `genofilename` | `str`  |         |          Path to `.geno` file.          |
| `snpfilename`  | `str`  |         |          Path to `.snp` file.           |
|    `chrNbr`    | `int`  |         |  Chromosome number to extract (1–24).   |
| `indfilename`  | `str`  |  `""`   |     Path to `.ind` file (optional).     |
|  `transpose`   | `bool` | `True`  | Whether to transpose the output matrix. |
|    `toCSV`     | `bool` | `False` |     Whether to export to CSV file.      |

**Returns:**

`df: pd.DataFrame`

* **df**: Genotype matrix of the specified chromosome.

**Notes:**

* Automatically filters target chromosome SNPs using `.snp` file.
* If an `.ind` file is provided, it defines the sample columns.
* When `chrNbr=24`, the function automatically calls `unpackYDNAfull()`.
* Optionally transposes the matrix or exports it to CSV.

**Example:**

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

Extract SNP data of a specified chromosome via `.anno` file.

**Parameters:**

|   Parameter    |  Type  | Default |               Description               |
|:--------------:|:------:|:-------:|:---------------------------------------:|
| `genofilename` | `str`  |         |          Path to `.geno` file.          |
| `snpfilename`  | `str`  |         |          Path to `.snp` file.           |
| `annofilename` | `str`  |         |          Path to `.anno` file.          |
|    `chrNbr`    | `int`  |         |        Target chromosome number.        |
|  `transpose`   | `bool` | `True`  | Whether to transpose the result matrix. |
|    `toCSV`     | `bool` | `False` |     Whether to export as CSV file.      |

**Returns:**

`df: pd.DataFrame`

* **df**: Genotype matrix of the specified chromosome (rows = SNPs, columns = samples).

**Notes:**

* Locates the target chromosome using `.snp` file.
* Uses `.anno` sample information to build column index.
* If chromosome = Y, it can use `unpackYDNAfromAnno()`.
* Supports transposing or exporting to CSV.

**Example:**

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

Filter Y chromosome sample indexes, retaining only valid male individuals.

**Parameters:**

|    Parameter     |        Type         |       Default       |                    Description                     |
|:----------------:|:-------------------:|:-------------------:|:--------------------------------------------------:|
|     `pdAnno`     |   `pd.DataFrame`    |                     |         DataFrame read from `.anno` file.          |
| `includefilters` | `list[str] \| None` |       `None`        | List of haplogroup keywords to include (optional). |
| `excludefilters` | `list[str] \| None` | `["na", " ", ".."]` |      List of haplogroup keywords to exclude.       |

**Returns:**

`malesId: list[int]`

* **malesId**: List of male sample indexes to retain.

**Notes:**

* If `includefilters` is specified, only haplogroups matching the list are retained.
* By default, excludes unknown haplogroups such as `na`, space, or `..`.
* Commonly used as a helper function in Y chromosome analysis.

**Example:**

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

Extract Y chromosome haplogroup information from `.anno` file.

**Parameters:**

|    Parameter     |        Type         | Default |               Description                |
|:----------------:|:-------------------:|:-------:|:----------------------------------------:|
|    `annofile`    |        `str`        |         |          Path to `.anno` file.           |
|   `separator`    |        `str`        | `"\t"`  | Delimiter of `.anno` file (default tab). |
| `includefilters` | `list[str] \| None` | `None`  |     Haplogroup keywords to include.      |
| `excludefilters` | `list[str] \| None` | `None`  |     Haplogroup keywords to exclude.      |

**Returns:**

`(ygroups: pd.Series, malesId: List[int])`

* **ygroups**: Series of selected haplogroups.
* **malesId**: Corresponding sample index list.

**Notes:**

* Relies on `FilterYhaplIndexes()` for gender and haplogroup filtering.
* Flexible configuration of include/exclude filters.
* Commonly used for data preparation before Y chromosome analysis.

**Example:**

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

Extract Y chromosome SNP genotype data based on `.anno` file.

**Parameters:**

|    Parameter     |        Type         | Default |             Description             |
|:----------------:|:-------------------:|:-------:|:-----------------------------------:|
|  `genofilename`  |        `str`        |         |        Path to `.geno` file.        |
|  `snpfilename`   |        `str`        |         |        Path to `.snp` file.         |
|  `annofilename`  |        `str`        |         |        Path to `.anno` file.        |
| `includefilters` | `list[str] \| None` | `None`  |   Haplogroup keywords to include.   |
| `excludefilters` | `list[str] \| None` | `None`  |   Haplogroup keywords to exclude.   |
|   `transpose`    |       `bool`        | `True`  | Whether to transpose result matrix. |
|     `toCSV`      |       `bool`        | `False` |   Whether to export to CSV file.    |

**Returns:**

`df: pd.DataFrame`

* **df**: Y chromosome SNP genotype matrix (rows = SNPs, columns = samples).

**Notes:**

* Automatically filters SNPs with `chromosome = 24` from `.snp` file.
* Uses `FilterYhaplIndexes()` to filter male samples and selected haplogroups.
* Supports transposing or exporting results to CSV.

**Example:**

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
## 4. io.py – Data Reading and Merging Interfaces

---

Encapsulates common methods for reading and saving CSV/table data.

### 📋 Function Overview

| Function Name |                     Description                     |
|:-------------:|:---------------------------------------------------:|
|  `load_geno`  |             Read genotype matrix (CSV).             |
|  `load_meta`  |            Read sample annotation table.            |
|  `load_csv`   | General CSV loading function (with error handling). |
|  `save_csv`   |            Export DataFrame as CSV file.            |

---

### 4.1 load_geno

Read genotype matrix.

**Parameters:**

| Parameter |      Type      |    Default     |      Description       |            |
|:---------:|:--------------:|:--------------:|:----------------------:|-----------:|
|  `path`   | `str  \| Path` |                |       File path.       |
| `id_col`  |     `str`      | `"Genetic ID"` | Sample ID column name. |            |
|   `sep`   |     `str`      |     `";"`      |       Separator.       |            |

**Returns:**

`(ids: pd.Series, X: pd.DataFrame, snp_cols: List[str])`

* **ids**: Sample ID series.
* **X**: SNP value matrix (rows = samples, columns = SNPs).
* **snp_cols**: List of SNP column names.

**Example:**

```python
import ancient_dna as adna
ids, X, snps = adna.load_geno("data/geno.csv")
```

---

### 4.2 load_meta

Read sample annotation table.

**Parameters:**

| Parameter |     Type      |    Default     |      Description       |            
|:---------:|:-------------:|:--------------:|:----------------------:|
|  `path`   | `str\|  Path` |                |       File path.       |
| `id_col`  |     `str`     | `"Genetic ID"` | Sample ID column name. |            
|   `sep`   |     `str`     |     `";"`      |       Separator.       |            

**Returns:**

`(meta: pd.DataFrame)`

* **meta**: Sample annotation table.

**Example:**

```python
import ancient_dna as adna
meta = adna.load_meta("data/meta.csv")
```

---

### 4.3 load_csv

General CSV loading function.

**Parameters:**

| Parameter |      Type       | Default | Description |            
|:---------:|:---------------:|:-------:|:-----------:|
|  `path`   | `str  \|  Path` |         | File path.  |
|   `sep`   |      `str`      |  `";"`  | Separator.  |

**Returns:**

`(df: pd.DataFrame)`

* **df**: Loaded DataFrame.

**Example:**

```python
import ancient_dna as adna
meta = adna.load_csv("data/demo.csv")
```

---

### 4.4 save_csv

Export DataFrame as a CSV file.

**Parameters:**

| Parameter |      Type       | Default |             Description             |            
|:---------:|:---------------:|:-------:|:-----------------------------------:|
|   `df`    | `pd.DataFrame`  |         |        DataFrame to export.         |            
|  `path`   | `str  \|  Path` |         |             File path.              |
|   `sep`   |      `str`      |  `";"`  |             Separator.              |            
|  `index`  |     `bool`      | `False` | Whether to include DataFrame index. |            
| `verbose` |     `bool`      | `True`  |     Whether to print save info.     |            

**Returns:**

`None`

**Example:**

```python
import pandas as pd
import ancient_dna as adna

X: pd.DataFrame = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})

adna.save_csv(X, "/geno_out.csv")
```

---

## 5. preprocess.py – Data Preprocessing and Missing Value Imputation

---

Provides data alignment, missing rate computation, and multiple imputation methods.

### 📋 Function Overview

|      Function Name      |                      Description                       |
|:-----------------------:|:------------------------------------------------------:|
|      `align_by_id`      |       Align sample IDs, keeping shared samples.        |
| `compute_missing_rates` |      Compute missing rates for samples and SNPs.       |
|   `filter_by_missing`   | Filter samples/SNPs exceeding missing-rate thresholds. |
|    `impute_missing`     |      Unified missing-value imputation interface.       |
|  `grouped_imputation`   |         Perform grouped imputation by labels.          |

---

### 5.1 align_by_id

Align sample IDs, keeping shared samples.

**Parameters:**

| Parameter |      Type      |    Default     |              Description              |
|:---------:|:--------------:|:--------------:|:-------------------------------------:|
|   `ids`   |  `pd.Series`   |                |           Sample ID series.           |
|    `X`    | `pd.DataFrame` |                |           Genotype matrix.            |
|  `meta`   | `pd.DataFrame` |                |            Metadata table.            |
| `id_col`  |     `str`      | `"Genetic ID"` | Column name of sample ID in metadata. |

**Returns:**

`(X_aligned: pd.DataFrame, meta_aligned: pd.DataFrame)`

* **X_aligned**: Genotype matrix retaining shared samples.
* **meta_aligned**: Metadata aligned with `X_aligned` rows.

**Example:**

```python
import pandas as pd
import ancient_dna as adna

ids: pd.Series = pd.Series(["A", "B", "A", "D"])
X: pd.DataFrame = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})
meta: pd.DataFrame = pd.DataFrame({
    "Genetic ID": ["A", "B", "A", "D"],
    "Y haplogroup": [2, 321, 12312, 421]
})
X1, meta1 = adna.align_by_id(ids, X, meta)
```

---

### 5.2 compute_missing_rates

Compute missing rates (sample-wise & SNP-wise).

* 0 = Reference allele.
* 1 = Variant allele.
* 3 = Missing value.

**Parameters:**

| Parameter |      Type      | Default |   Description    |
|:---------:|:--------------:|:-------:|:----------------:|
|    `X`    | `pd.DataFrame` |         | Genotype matrix. |

**Returns:**

`(sample_missing: pd.Series, snp_missing: pd.Series)`

* **sample_missing**: Missing rate for each sample (row).
* **snp_missing**: Missing rate for each SNP (column).

**Example:**

```python
import pandas as pd
import ancient_dna as adna

X1: pd.DataFrame = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})
sm, cm = adna.compute_missing_rates(X1)
```

---

### 5.3 filter_by_missing

Filter samples and SNPs by missing rate thresholds.

**Parameters:**

|      Parameter       |      Type      | Default |           Description            |
|:--------------------:|:--------------:|:-------:|:--------------------------------:|
|         `X`          | `pd.DataFrame` |         |         Genotype matrix.         |
|   `sample_missing`   |  `pd.Series`   |         |     Missing rate per sample.     |
|    `snp_missing`     |  `pd.Series`   |         |      Missing rate per SNP.       |
| `max_sample_missing` |    `float`     |  `0.8`  | Max allowed sample missing rate. |
|  `max_snp_missing`   |    `float`     |  `0.8`  |  Max allowed SNP missing rate.   |

**Returns:**

`(X_filtered: pd.DataFrame)`

* **X_filtered**: Filtered matrix.

**Example:**

```python
import pandas as pd
import ancient_dna as adna

X1: pd.DataFrame = pd.DataFrame({
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

Perform missing-value imputation.

**Parameters:**

|   Parameter   |      Type      | Default  |                                                               Description                                                                |
|:-------------:|:--------------:|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------:|
|      `X`      | `pd.DataFrame` |          |                                                             Genotype matrix.                                                             |
|   `method`    |     `str`      | `"mode"` | Imputation method (`mode`, `mean`, `knn`, `knn_hamming`, `knn_hamming_abs`, `knn_hamming_adaptive`, `knn_hybrid_autoalpha`, `knn_auto`). |
| `n_neighbors` |     `int`      |   `5`    |                                                 Number of neighbors for KNN imputation.                                                  |

**Returns:**

`(filled: pd.DataFrame)`

* **filled**: Imputed matrix.

**Example:**

```python
import pandas as pd
import ancient_dna as adna

X: pd.DataFrame = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})
filled = adna.impute_missing(X, method="knn")
```

---

### 5.5 grouped_imputation

Perform grouped missing-value imputation (wrapped version).
Based on external labels (e.g., geographic regions or haplogroups), the dataset is divided into subsets, and imputation is performed separately within each group.
If no labels are provided, a global imputation is applied.

**Parameters:**

| Parameter |        Type         | Default  |                                          Description                                          |
|:---------:|:-------------------:|:--------:|:---------------------------------------------------------------------------------------------:|
|    `X`    |   `pd.DataFrame`    |          |                  Original genotype matrix (rows = samples, columns = SNPs).                   |
| `labels`  | `pd.Series \| None` |          | External grouping labels (e.g., region or haplogroup). If `None`, performs global imputation. |
| `method`  |        `str`        | `"mode"` |       Missing-value imputation method (e.g., `"mode"`, `"knn_hamming_adaptive"`, etc.).       |

**Returns:**

`(filled_X: pd.DataFrame)`

* **filled_X**: Fully imputed matrix, index order consistent with the original matrix.

**Algorithm Steps:**

1. If `labels=None`, perform global imputation directly.
2. Otherwise, divide samples into subsets by label.
3. Apply `impute_missing()` separately to each group.
4. For small groups (≤5 samples), fallback to column-mode imputation.
5. If method is `"knn_faiss"` and sample count is too small, fallback to `"mode"`.
6. Merge all results and restore original index order.

**Example:**

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

## 6. summary.py – Data Analysis and Summary Tools

---

This module is used to automatically generate data analysis reports, dimensionality reduction statistics, and runtime summary tables.

### 📋 Function Overview

|      Function Name       |                     Description                      |
|:------------------------:|:----------------------------------------------------:|
|  `build_missing_report`  |         Generate missing rate summary table.         |
| `build_embedding_report` | Compute statistical summaries for embedding results. |
|      `save_report`       |            Save report table as CSV file.            |
|  `save_runtime_report`   |      Save runtime summary table of algorithms.       |

---

### 6.1 build_missing_report

Generate summary table for sample and SNP missing rates.

**Parameters:**

|    Parameter     |    Type     | Default |          Description          |
|:----------------:|:-----------:|:-------:|:-----------------------------:|
| `sample_missing` | `pd.Series` |         | Missing rate for each sample. |
|  `snp_missing`   | `pd.Series` |         |  Missing rate for each SNP.   |

**Returns:**

`df: pd.DataFrame`

* A single-row statistical report containing mean, median, and maximum values.
* The table includes the following fields:

  * **sample_count**: Total number of samples.
  * **snp_count**: Total number of SNPs.
  * **sample_missing_mean**: Mean missing rate of samples.
  * **sample_missing_median**: Median missing rate of samples.
  * **sample_missing_max**: Maximum missing rate of samples.
  * **snp_missing_mean**: Mean missing rate of SNPs.
  * **snp_missing_median**: Median missing rate of SNPs.
  * **snp_missing_max**: Maximum missing rate of SNPs.

**Example:**

```python
import pandas as pd
import ancient_dna as adna

sample_missing: pd.Series = pd.Series([0.7, 0.2, 0.4, 0.1])
snp_missing: pd.Series = pd.Series([0.7, 0.2, 0.4, 0.1])
report = adna.build_missing_report(sample_missing, snp_missing)
```

---

### 6.2 build_embedding_report

Generate statistical report for embedding results.

**Parameters:**

|  Parameter  |      Type      | Default |                                          Description                                          |
|:-----------:|:--------------:|:-------:|:---------------------------------------------------------------------------------------------:|
| `embedding` | `pd.DataFrame` |         | Dimensionality reduction result DataFrame, usually with columns like `['Dim1', 'Dim2', ...]`. |

**Returns:**

`pd.DataFrame`

* Contains mean, standard deviation, minimum, and maximum values for each dimension.

  * **Dimension**: Dimension name.
  * **Mean**: Mean value.
  * **StdDev**: Standard deviation.
  * **Min**: Minimum value.
  * **Max**: Maximum value.

**Example:**

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

Save the report table as a CSV file.

**Parameters:**

| Parameter |      Type      | Default |     Description      |            
|:---------:|:--------------:|:-------:|:--------------------:|
|   `df`    | `pd.DataFrame` |         | DataFrame to export. |            
|  `path`   | `str \|  Path` |         |      File path.      |

**Returns:**

`None`

**Example:**

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

Save runtime summary table for imputation and embedding algorithms.

**Parameters:**

| Parameter |      Type      | Default |                                                    Description                                                    |                                         
|:---------:|:--------------:|:-------:|:-----------------------------------------------------------------------------------------------------------------:|
| `records` |  `list[dict]`  |         | List of runtime records. Example:`[{"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52}]` |                                         
|  `path`   | `str \|  Path` |         |                                      Output file path (including file name).                                      |  

**Returns:**

`None`

**Example:**

```python
import ancient_dna as adna

records = [
    {"imputation_method": "mode", "embedding_method": "umap", "runtime_s": 6.52},
    {"imputation_method": "mean", "embedding_method": "pca", "runtime_s": 1.84}
]
adna.save_runtime_report(records, "data/results/runtime_summary.csv")
```

---

## 7. visualize.py – Visualization Tools

---

This module is used to plot dimensionality reduction scatter plots, missing-data distributions, and other analytical figures.

### 📋 Function Overview

|        Function Name        |                              Description                               |
|:---------------------------:|:----------------------------------------------------------------------:|
|      `plot_embedding`       |               Plot 2D dimensionality reduction results.                |
|    `plot_missing_values`    |              Visualize missing data distribution matrix.               |
| `plot_cluster_on_embedding` | Overlay cluster results on embedding and show dominant label & purity. |
|   `plot_silhouette_trend`   | Plot relationship between cluster number and average silhouette score. |

---

### 7.1 plot_embedding

Plot 2D embedding scatter plot with customizable legend and color mapping.
Categories beyond `legend_max` are displayed in gray in both the plot and legend.

**Parameters:**

|   Parameter    |         Type         |        Default         |                        Description                         |                                       
|:--------------:|:--------------------:|:----------------------:|:----------------------------------------------------------:| 
|      `df`      |    `pd.DataFrame`    |                        | Dimensionality reduction result containing `Dim1`, `Dim2`. |                                       
|    `labels`    |     `pd.Series`      |                        |              Optional classification labels.               |                                       
|    `title`     |        `str`         |                        |                        Plot title.                         |                                       
|  `save_path`   | `str\| Path \| None` |                        |           Save path (display directly if None).            |
|   `figsize`    |       `tuple`        |       `(10, 7)`        |                        Figure size.                        |                                       
|  `legend_pos`  |        `str`         |       `"right"`        |  Legend position: `right`, `bottom`, `top`, or `inside`.   |                                       
|     `cmap`     |        `str`         |       `"tab20"`        |                         Colormap.                          |                                       
|  `legend_max`  |        `int`         |          `20`          |          Maximum number of categories to display.          |                                       
| `legend_sort`  |        `bool`        |         `True`         |              Whether to sort by sample count.              |                                      
| `others_color` |       `tuple`        | `(0.7, 0.7, 0.7, 0.5)` |           Color for samples beyond legend limit.           |                                       

**Returns:**

`None`

**Example:**

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

Smart visualization of missing data patterns. Automatically switches between detailed pixel visualization (for small matrices) and aggregated missing-rate histograms (for large matrices).

**Parameters:**

|    Parameter    |          Type          |   Default   |                              Description                              |                                       
|:---------------:|:----------------------:|:-----------:|:---------------------------------------------------------------------:| 
|      `df`       |     `pd.DataFrame`     |             |                         Genotype/sample data.                         |                                       
|   `save_path`   | `str  \| Path  \|None` |             |                 Save path (display directly if None).                 |
| `missing_value` |         `int`          |     `3`     |                         Missing-value marker.                         |                                       
|    `figsize`    |        `tuple`         |  `(10, 7)`  |                             Figure size.                              |                                       
| `cmap_present`  |         `str`          | `"#d95f02"` |                       Color for present values.                       |                                       
| `cmap_missing`  |         `str`          | `"#ffffff"` |                       Color for missing values.                       |                                       
|  `show_ratio`   |         `bool`         |   `True`    |                 Whether to display missing-rate bars.                 |                                       
|  `max_pixels`   |         `int`          |    `5e7`    | When total pixels exceed this threshold, switches to aggregated mode. |                                       

**Returns:**

`None`

**Example:**

```python
import pandas as pd
import ancient_dna as adna

X: pd.DataFrame = pd.DataFrame({
    "SNP1": [0, 1, 3, 1],
    "SNP2": [1, 3, 0, 1],
    "SNP3": [3, 3, 1, 0]
})

adna.plot_missing_values(X, "results/missing_values.png")
```

---

### 7.3 plot_cluster_on_embedding

Overlay clustering results on embedding visualization.
Each cluster is annotated with its **Dominant Label** and **Purity (Dominant %)** at the cluster center, providing an intuitive way to evaluate clustering quality and label consistency.

**Parameters:**

|   Parameter    |          Type           |             Default             |                           Description                           |                                                                 
|:--------------:|:-----------------------:|:-------------------------------:|:---------------------------------------------------------------:|
| `embedding_df` |     `pd.DataFrame`      |                                 |          Embedding result with `Dim1`, `Dim2` columns.          |                                                                 
|    `labels`    |       `pd.Series`       |                                 |                 Cluster labels, one per sample.                 |                                                                 
|     `meta`     | `pd.DataFrame \| None`  |             `None`              | Optional metadata used for computing dominant label and purity. |
|  `label_col`   |          `str`          |         `"World Zone"`          |        Metadata label column for evaluating consistency.        |                                                                 
|    `title`     |          `str`          | `"Clusters on Embedding Space"` |                           Plot title.                           |                                                                 
|   `figsize`    |         `tuple`         |            `(8, 6)`             |                          Figure size.                           |                                                                 
|  `save_path`   |     `Path \| None`      |             `None`              |          Save path (if provided) or display directly.           |

**Returns:**

None (Displays or saves the plot.)

**Algorithm Steps:**

1. Verify that `Dim1` and `Dim2` exist in the input data.
2. Plot scatter points for clusters using distinct colors.
3. If metadata is provided:

   * Compute cluster center coordinates.
   * Count real labels within each cluster.
   * Determine dominant label and purity (%).
   * Annotate each cluster center with dominant label and purity.
4. Output a publication-quality visualization (either saved or displayed).

**Example:**

```python
import pandas as pd
from pathlib import Path
from ancient_dna import plot_cluster_on_embedding

# Simulated embedding and clustering results
embedding = pd.DataFrame({
    "Dim1": [0.1, 0.3, 0.8, 1.0],
    "Dim2": [0.2, 0.5, 0.9, 1.2]
})
labels = pd.Series([0, 0, 1, 1])
meta = pd.DataFrame({
    "World Zone": ["Europe", "Europe", "Asia", "Asia"]
})

# Plot and save
plot_cluster_on_embedding(
    embedding_df=embedding,
    labels=labels,
    meta=meta,
    label_col="World Zone",
    title="Example: Cluster Overlay Plot",
    save_path=Path("results/cluster_plot.png")
)
```

**Notes:**

* Point colors represent different clusters.
* If metadata is provided, purity of dominant labels per cluster is calculated.
* Small or mixed clusters will exhibit lower purity.
* The result can be used to assess clustering quality and label consistency.
* If no save path is specified, the plot is shown directly.

---

### 7.4 plot_silhouette_trend

Plot the **Silhouette Score** trend as the number of clusters changes.
This helps determine the optimal number of clusters (k) by visualizing silhouette scores for different k values to assess clustering quality and stability.

**Parameters:**

|  Parameter  |           Type            | Default |                                                    Description                                                    |                                                                
|:-----------:|:-------------------------:|:-------:|:-----------------------------------------------------------------------------------------------------------------:|
|  `scores`   | `list[tuple[int, float]]` |         | List of tuples representing cluster numbers and their silhouette scores, each element as `(k, silhouette_score)`. |                                                                
| `save_path` |     `Path  \|  None`      | `None`  |                          Save path if provided; otherwise, display the figure directly.                           |

**Returns:**

None (Displays or saves the trend plot.)

**Algorithm Steps:**

1. Extract cluster numbers and corresponding silhouette scores from the `(k, score)` list.
2. Plot a line chart with the x-axis as the cluster number `k` and y-axis as the silhouette score.
3. Automatically set grid, title, and axis labels.
4. Save the figure if `save_path` is provided.
5. Otherwise, display it on screen.

**Example:**

```python
from pathlib import Path
from ancient_dna import plot_silhouette_trend

# Simulated silhouette scores for different cluster numbers
scores = [(2, 0.41), (3, 0.52), (4, 0.49), (5, 0.45), (6, 0.43)]

# Plot and save
plot_silhouette_trend(scores, save_path=Path("results/silhouette_trend.png"))
```

**Notes:**

* Higher **Silhouette Scores** indicate clearer cluster separation and better structure.
* Typically, the **k value with the highest score** is chosen as the optimal cluster number.
* This plot is suitable for evaluating clustering models such as KMeans and Agglomerative Clustering.

---

## 📚 Appendix

---

#### A. Common Terms and Abbreviations

|    Term / Abbreviation    |                                  Description                                   |
|:-------------------------:|:------------------------------------------------------------------------------:|
|          **SNP**          |                        Single Nucleotide Polymorphism.                         |
| **`.geno`, `.snp`, etc.** |                     EIGENSTRAT-format genotype data files.                     |
|       **Embedding**       |   Mapping high-dimensional genotype data into 2D/3D space for visualization.   |
|    **Missing Values**     | Missing or unidentifiable alleles in data, represented by `3` in this project. |
|      **Haplogroup**       |           Evolutionary lineage of Y-chromosome or mitochondrial DNA.           |

---

#### B. Error and Exception Reference

|   Exception Type    |                          Possible Cause                          |                  Suggested Solution                   |
|:-------------------:|:----------------------------------------------------------------:|:-----------------------------------------------------:|
| `FileNotFoundError` |       Missing or incorrect file path (e.g., `load_csv()`).       |            Verify file path and existence.            |
|    `ValueError`     |     Invalid argument values (e.g., misspelled method names).     |      Check if method name matches valid options.      |
|     `KeyError`      |             Accessing non-existent DataFrame column.             | Ensure correct ID column name (e.g., `"Genetic ID"`). |
|   `RuntimeError`    |   Runtime failure (e.g., corrupted or malformed input files).    |    Validate data formatting; debug incrementally.     |
|     `TypeError`     | Incorrect argument type (e.g., passing non-`DataFrame` objects). |     Ensure correct types such as `pd.DataFrame`.      |

---

#### C. File Format Reference (EIGENSTRAT)

|      File Type      | Extension |                         Description                         |
|:-------------------:|:---------:|:-----------------------------------------------------------:|
|   Genotype Matrix   |  `.geno`  | Encoded as 0, 1, 3 representing alleles and missing values. |
|   SNP Information   |  `.snp`   |    Each row describes SNP position and chromosome info.     |
| Individual Metadata |  `.ind`   |         Contains gender and population information.         |
| Annotation Metadata |  `.anno`  |   Includes geographic, temporal, and haplogroup metadata.   |

---

### D. Version History

|  Version   |    Date    |                                         Description                                         |
|:----------:|:----------:|:-------------------------------------------------------------------------------------------:|
| **v0.1.0** | 2025-10-16 | First release of API documentation, including core modules like embedding and genotopython. |
| **v0.1.1** | 2025-10-24 |             Fixed return name errors, incorrect example calls, and minor typos.             |
| **v0.2.0** | 2025-11-08 |             Added clustering module and functions optimized for large datasets.             |





