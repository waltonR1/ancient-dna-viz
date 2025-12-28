# 🧬 ancient-dna-viz

> 📄 中文版本请见：[README.zh-CN.md](README.zh-CN.md)

## A Toolchain for Ancient DNA Genotype Analysis, Dimensionality Reduction, Clustering, and Visualization

---

## 1. Background and Research Motivation

In ancient DNA (aDNA) research, practitioners typically face the following data characteristics and challenges:

1. **High dimensionality**
   A single sample often contains hundreds of thousands to millions of SNP loci;

2. **Extremely high missing rates**
   Due to DNA degradation and low sequencing coverage, aDNA datasets exhibit substantially higher missingness than modern genomic data;

3. **Strong reliance on exploratory analysis**
   Many research questions are not formulated as supervised learning problems, but instead require interpretation based on geometric structure, clustering patterns, and existing archaeological or geographical knowledge;

4. **Coexistence of large data scale and limited memory resources**
   In personal workstations or teaching environments, loading and processing the full genotype matrix at once is often infeasible.

As a result, ancient DNA analysis is rarely an “end-to-end automated modeling” problem. Instead, it is better understood as a process focused on **data processing, structural exposure, and interpretability support**.

---

## 2. Installation and Dependencies

This project is intended to be used **from source**.
This section only describes the Python dependency environment required for running and developing the codebase.

### 2.1 Python Environment Requirements

* Python ≥ 3.10
* The use of a virtual environment (`venv` / `conda`) is recommended

### 2.2 Dependency Installation

From the project root directory, run:

```bash
pip install -r requirements.txt
pip install -e .
```

Where:

* `requirements.txt` defines all third-party dependencies required for running and analysis;
* `pip install -e .` installs `ancient_dna` in editable (development) mode, which is suitable for research and script-based workflows and allows source code changes to take effect without reinstallation.

---

## 3. Project Objectives and Design Boundaries

### 3.1 Project Objectives

The goal of `ancient-dna-viz` is to provide a toolchain that is:

* **Interpretable**
* **Reproducible**
* **Extensible**

and that supports the following tasks:

* Preprocessing of genotype data and missing-rate analysis
* Multiple missing-value imputation strategies under high missingness
* Low-dimensional representations of high-dimensional SNP matrices
* Clustering structure analysis in reduced embedding spaces
* Consistency evaluation between clustering results and known labels (e.g., geography or haplogroups)
* Static and interactive visualization of analysis results

---

### 3.2 Explicit Scope Exclusions

To preserve methodological transparency and engineering maintainability, this project deliberately **does not include**:

* Automatic generation of genetic conclusions
* End-to-end “black-box” AI decision models
* Implicit prediction of population membership
* Hidden data filtering or automatic label correction

All important data transformations are exposed through **explicit function calls**.

---

## 4. Methodological Overview

The overall analysis workflow can be abstracted into the following stages:

```text
Raw genotype data
  ↓
Sample / SNP alignment
  ↓
Missing-rate analysis
  ↓
Threshold-based filtering
  ↓
Missing-value imputation
  ↓
Dimensionality reduction (embedding)
  ↓
Clustering analysis
  ↓
Clustering–label consistency evaluation
  ↓
Visualization and result export
```

Each stage is handled by an independent module, avoiding cross-layer implicit dependencies and improving debuggability and interpretability.

---

## 5. Project Structure and Module Responsibilities

```text
ancient-dna-viz/
├── data/
│   ├── raw/        # Raw input data
│   ├── processed/  # Intermediate data (e.g., imputed matrices)
│   └── results/    # Outputs (figures / CSV / HTML)
│
├── docs/
│   ├── API_REFERENCE_CN.md
│   └── API_REFERENCE_EN.md
│
├── scripts/
│   ├── preprocess_csv.py
│   ├── preprocess_geno.py
│   └── inspect_ancestrymap.py
│
├── src/ancient_dna/
│   ├── preprocess.py
│   ├── embedding.py
│   ├── clustering.py
│   ├── visualize.py
│   ├── io.py
│   ├── genotopython.py
│   ├── summary.py
│   └── tests/
│
├── README.md
└── README.zh-CN.md
```

---

## 6. Module Design Overview

### 6.1 Data Preprocessing Module (`preprocess.py`)

This module is responsible for all **structural operations at the sample and SNP dimensions**, including:

* Sample ID alignment (genotype matrix ↔ metadata)
* Sample- and SNP-level missing-rate computation
* Threshold-based filtering strategies
* Multiple missing-value imputation methods

Design principles:

* All filtering operations **explicitly return new objects**
* No implicit in-place modification of input data
* Clear missing-value semantics (e.g., `3` denotes missing)

---

### 6.2 Dimensionality Reduction Module (`embedding.py`)

Provides a unified interface `compute_embeddings`, supporting:

* UMAP
* t-SNE
* MDS
* Isomap

For large-scale SNP data, the module additionally supports:

* **Pseudo-streaming UMAP based on Incremental PCA**
* Parquet shard loading to control memory usage

This module focuses exclusively on **geometric representations**, without introducing labels or biological interpretation.

---

### 6.3 Clustering Module (`clustering.py`)

Clustering analysis is based on **agglomerative (hierarchical) clustering**, for the following reasons:

* No reliance on random initialization
* Clear hierarchical structure
* Well-suited for exploratory analysis

Supported features include:

* Automatic search for the optimal number of clusters (Silhouette Score)
* Clustering in the original high-dimensional space
* Clustering in reduced embedding spaces
* Consistency evaluation between clustering results and known labels (purity analysis)

---

### 6.4 Visualization Module (`visualize.py`)

Visualization functionality is divided into three categories:

1. **Structural visualizations**

   * Embedding scatter plots
   * Missing-value distribution plots

2. **Clustering interpretation visualizations**

   * Cluster overlays
   * Dominant label and purity annotations

3. **Evaluation support visualizations**

   * Silhouette score trends

Both of the following backends are supported:

* Matplotlib (publication-quality static figures)
* Plotly (interactive exploration)

---

## 7. Role of the `scripts/` Directory

The `scripts/` directory contains **dataset-specific analysis workflows**, rather than library code, such as:

* Reading and preprocessing `.geno / .snp / .ind` files
* Batch execution of full analysis pipelines
* Structural inspection of specific datasets

These scripts are:

* Modifiable
* Oriented toward specific datasets
* Not part of the public API

---

## 8. Intended Usage Pattern

The recommended usage pattern for this project is:

* Use `ancient_dna` as an **analysis toolbox**
* Explicitly compose workflows in `scripts/` or notebooks
* Preserve intermediate results for traceability and interpretation

Rather than:

* Producing conclusions via a single command
* Hiding data cleaning and filtering steps

---

## 9. Reproducibility and Engineering Considerations

* All stochastic processes support `random_state`
* Critical parameters avoid implicit “magic defaults”
* Results can be exported as CSV / PNG / HTML
* Unit tests are provided to ensure stable core behavior

---

## 10. Documentation and Testing

* 📘 Chinese API documentation: [`docs/API_REFERENCE_CN.md`](docs/API_REFERENCE_CN.md)
* 📘 English API documentation: [`docs/API_REFERENCE_EN.md`](docs/API_REFERENCE_EN.md)
* 🧪 Testing framework: `pytest`

---

## 11. Project Summary

`ancient-dna-viz` is a **method-driven, interpretation-first** toolkit for ancient DNA data analysis.
Its value lies not in automated prediction, but in:

> **Transforming complex genetic structures into observable, discussable, and verifiable geometric and statistical patterns.**