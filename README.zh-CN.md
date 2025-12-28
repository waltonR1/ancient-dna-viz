# 🧬 ancient-dna-viz

> 📄 English version: [README.md](README.md)

## 古 DNA 基因型数据分析、降维、聚类与可视化工具链

---

## 1. 项目背景与研究动机

在古 DNA（ancient DNA, aDNA）研究中，研究者通常面对以下数据特征与挑战：

1. **高维度**
   单个样本往往包含数十万至上百万个 SNP 位点；

2. **极高缺失率**
   由于 DNA 降解、测序覆盖不足等原因，古 DNA 数据中缺失值比例显著高于现代基因组数据；

3. **强依赖探索性分析**
   很多研究问题并不存在明确的监督标签，而是需要通过几何结构、聚类模式与已有考古 / 地理知识进行解释；

4. **数据规模与内存限制并存**
   在个人工作站或教学环境中，难以一次性加载完整矩阵进行计算。

因此，古 DNA 数据分析往往不是一个“端到端自动建模”的问题，而是一个 **数据处理、结构暴露与解释支持** 的过程。

---

## 2. 安装与依赖（Installation & Dependencies）

本项目默认以源码形式使用，以下仅说明运行与开发所需的 Python 依赖环境。

### 2.1 Python 环境要求

- Python ≥ 3.10
- 推荐使用虚拟环境（venv / conda）

### 2.2 依赖库安装

在项目根目录下执行：
```bash
pip install -r requirements.txt
pip install -e .
```

**说明：**

- `requirements.txt`
安装项目运行与分析所需的第三方依赖库；

- `pip install -e .`
以 editable（开发）模式安装 ancient_dna，
适合科研与脚本式分析流程，修改源码后无需重新安装。

---

## 3. 项目目标与设计边界

### 3.1 项目目标

`ancient-dna-viz` 的目标是提供一套：

* **可解释（interpretable）**
* **可复现（reproducible）**
* **可扩展（extensible）**

的工具链，用于支持以下任务：

* 基因型数据的预处理与缺失分析
* 高缺失率条件下的多策略缺失值填补
* 高维 SNP 矩阵的降维表达
* 降维空间中的聚类结构分析
* 聚类结果与已知标签（地理 / 单倍群等）的一致性评估
* 结果的静态与交互式可视化展示

---

### 3.2 明确不做的事情（Scope Exclusions）

为了保持方法透明性与工程可维护性，本项目**刻意不包含**以下内容：

*  自动遗传学结论生成
*  端到端“AI 决策模型”
*  群体归属的黑箱预测
*  隐式数据过滤或自动标签修正

所有重要的数据变换步骤均以**显式函数调用**的方式呈现。

---

## 4. 方法论概览（Methodological Overview）

整体分析流程可抽象为如下阶段：

```text
原始基因型数据
  ↓
样本 / SNP 对齐
  ↓
缺失率分析
  ↓
阈值过滤
  ↓
缺失值填补
  ↓
降维表示（Embedding）
  ↓
聚类分析
  ↓
聚类与真实标签一致性评估
  ↓
可视化与结果导出
```

每一阶段均由独立模块负责，避免跨层隐式依赖，提高可调试性与可解释性。

---

## 5. 项目结构与模块职责

```text
ancient-dna-viz/
├── data/
│   ├── raw/        # 原始输入数据
│   ├── processed/  # 中间数据（填补后矩阵等）
│   └── results/    # 输出结果（图像 / CSV / HTML）
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

## 6. 各模块设计说明

### 6.1 数据预处理模块（`preprocess.py`）

该模块承担所有 **样本维度与 SNP 维度的结构性操作**，包括：

* 样本 ID 对齐（矩阵 ↔ 元数据）
* 样本 / SNP 缺失率计算
* 基于阈值的过滤策略
* 多种缺失值填补方法

设计原则：

* 所有过滤逻辑 **显式返回新对象**
* 不在函数内部隐式修改原始数据
* 缺失值语义明确（如 `3` 表示缺失）

---

### 6.2 降维模块（`embedding.py`）

提供统一的降维接口 `compute_embeddings`，支持：

* UMAP
* t-SNE
* MDS
* Isomap

并针对大规模 SNP 数据，提供：

* **基于 Incremental PCA 的伪流式 UMAP**
* Parquet 分片读取，避免内存峰值

该模块专注于 **几何表示**，不涉及标签或生物学解释。

---

### 6.3 聚类模块（`clustering.py`）

聚类分析基于 **层次聚类（Agglomerative Clustering）**，原因包括：

* 不依赖随机初始化
* 聚类结构层次清晰
* 适合探索性分析

支持：

* 自动搜索最优聚类数（Silhouette Score）
* 高维空间聚类
* 降维空间聚类
* 聚类与真实标签一致性评估（纯度分析）

---

### 6.4 可视化模块（`visualize.py`）

可视化模块分为三类：

1. **结构可视化**

   * embedding 散点图
   * 缺失值分布图

2. **聚类解释可视化**

   * 聚类叠加图
   * 主标签与纯度标注

3. **评估辅助可视化**

   * 轮廓系数趋势图

同时支持：

* Matplotlib（出版级静态图）
* Plotly（交互式探索）

---

## 7. scripts 目录的角色

`scripts/` 目录用于存放 **具体分析流程脚本**，而非库代码，例如：

* 从 `.geno / .snp / .ind` 读取并预处理数据
* 批量执行完整 pipeline
* 对特定数据集进行结构检查

这些脚本是：

*  可修改的
*  面向具体数据集的
*  不作为公共 API

---

## 8. 使用方式定位说明

本项目推荐的使用方式是：

* 将 `ancient_dna` 作为 **分析工具箱**
* 在 `scripts/` 或 Notebook 中 **显式组合流程**
* 保留中间结果，方便回溯与解释

而不是：

* 一行命令输出“结论”
* 隐藏数据清洗与过滤细节

---

## 9. 可复现性与工程考虑

* 所有随机过程支持 `random_state`
* 重要参数不使用魔法默认值
* 输出结果可保存为 CSV / PNG / HTML
* 提供单元测试以保证基础函数行为稳定

---

## 10. 文档与测试

* 📘 中文 API 文档：[`docs/API_REFERENCE_CN.md`](docs/API_REFERENCE_CN.md)
* 📘 英文 API 文档：[`docs/API_REFERENCE_EN.md`](docs/API_REFERENCE_EN.md)
* 🧪 测试框架：`pytest`

---

## 11. 项目总结

`ancient-dna-viz` 是一个 **方法导向、解释优先** 的古 DNA 数据分析工具库，其价值不在于“自动预测”，而在于：

> **将复杂的遗传结构转化为可观察、可讨论、可验证的几何与统计模式。**

