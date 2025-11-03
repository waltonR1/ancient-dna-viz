import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import multiprocessing

def align_by_id(ids: pd.Series, X: pd.DataFrame, meta: pd.DataFrame, id_col: str = "Genetic ID") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按样本 ID 对齐基因型矩阵与注释表，仅保留两表共有的样本，并保证行顺序一致。

    :param ids: 样本 ID 序列（与 X 的行一一对应）。
    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param meta: 注释表 (pd.DataFrame)，包含样本 ID 以及标签信息（如 Y haplogroup）。
    :param id_col: 注释表中的样本 ID 列名（默认 "Genetic ID"）。
    :return: (X_aligned, meta_aligned)
             - X_aligned: 仅保留共有样本后的基因型矩阵（重新从 0 开始索引）。
             - meta_aligned: 与 X_aligned 行顺序一致的注释表。
    注意:
        - 若注释表中缺失某些样本，将被自动丢弃；仅保留交集。
    """
    print("[INFO] Align by ID:")
    ids_unique = ids.drop_duplicates().reset_index(drop=True)
    meta_ids = meta[id_col].drop_duplicates()
    common_ids = ids_unique[ids_unique.isin(meta_ids)]

    mask = ids.isin(common_ids)
    X_aligned = X.loc[mask].reset_index(drop=True)
    meta_aligned = meta.set_index(id_col).loc[common_ids].reset_index()
    print(f"[OK] 共 {len(common_ids)} 个样本在两表中匹配 ({len(ids)}→{len(common_ids)})")
    return X_aligned, meta_aligned


def compute_missing_rates(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    计算缺失率（样本维度 & SNP 维度）。
        - 0 = 参考等位基因
        - 1 = 变异等位基因
        - 3 = 缺失

    :param X: 基因型矩阵 (pd.DataFrame)。
    :return: (sample_missing, snp_missing)
             - sample_missing: 每个样本（行）的缺失率 (0~1)。
             - snp_missing: 每个 SNP（列）的缺失率 (0~1)。
    """
    print("[INFO] Compute missing rate:")
    # 将编码 3 替换为 NaN 以便统计缺失率
    Z = X.replace(3, np.nan)
    # 计算每行（样本）与每列（SNP）的缺失率
    sample_missing = Z.isna().mean(axis=1)
    snp_missing = Z.isna().mean(axis=0)
    sample_missing.name = "sample_missing_rate"
    snp_missing.name = "snp_missing_rate"
    print("[OK] Computed the missing rate of each sample and SNP.")
    return sample_missing, snp_missing


def filter_by_missing(X: pd.DataFrame, sample_missing: pd.Series, snp_missing: pd.Series, max_sample_missing: float = 0.8, max_snp_missing: float = 0.8) -> pd.DataFrame:
    """
    按缺失率阈值过滤样本与 SNP（默认阈值较宽松，可在后续调整）。

    :param X: 基因型矩阵。
    :param sample_missing: 每个样本的缺失率 (pd.Series)。
    :param snp_missing: 每个 SNP 的缺失率 (pd.Series)。
    :param max_sample_missing: 样本级最大缺失率阈值（默认 0.8）。
    :param max_snp_missing: SNP 级最大缺失率阈值（默认 0.8）。
    :return: 过滤后的矩阵 (pd.DataFrame)，索引从 0 重新开始。
    """
    print("[INFO] Filter by missing rate:")
    keep_rows = sample_missing <= max_sample_missing
    keep_cols = snp_missing[snp_missing <= max_snp_missing].index
    X_filtered = X.loc[keep_rows, keep_cols].reset_index(drop=True)
    print(f"[INFO] 样本: {keep_rows.sum()}/{len(X)} 保留, SNP: {len(keep_cols)}/{X.shape[1]} 保留")

    if X_filtered.empty:
        raise ValueError("[ERROR] 过滤后矩阵为空，请调整阈值。")

    return X_filtered


def _fill_mode(Z: pd.DataFrame, fallback: float = 1) -> pd.DataFrame:
    """
    列众数填补（默认方法）
    ===================================================
    基于每列的众数（忽略 NaN）进行填补。

    :param Z: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :param fallback: float, default=1
        若整列均为空，则使用该值填补。
    :return: pd.DataFrame
        填补后的矩阵，与输入结构一致。

    说明:
        - 对每列分别计算众数（忽略 NaN）；
        - 若整列均为空，则使用 fallback；
        - 比 apply(axis=0) 实现更高效（避免逐列函数调度开销）。
        - 输出与原矩阵结构完全一致。
    """

    filled_cols = {}  # 用字典暂存所有填补结果
    for col in Z.columns:
        col_data = Z[col]
        if col_data.isna().all():
            mode_val = fallback
        else:
            mode_val = col_data.mode(dropna=True).iloc[0]
        filled_cols[col] = col_data.fillna(mode_val)

    # 一次性拼接，避免碎片化
    result = pd.concat(filled_cols, axis=1)
    result.columns = Z.columns
    result.index = Z.index
    return result


def _fill_mean(X: pd.DataFrame) -> pd.DataFrame:
    """
    列均值填补（矢量化实现）。
    ===================================================
    使用每列均值替代缺失值，适合连续型数据。

    :param X: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :return: pd.DataFrame
        填补后的矩阵。

    说明:
        - 对每列单独计算均值；
        - 用 fillna() 替换 NaN；
        - 输出与原矩阵列名一致。
    """
    # pandas fillna 本身矢量化，不会触发碎片警告
    result = X.fillna(X.mean())
    result.columns = X.columns
    result.index = X.index
    return result


def _fill_knn(X: pd.DataFrame, n_neighbors: int = 5, metric: str = "nan_euclidean") -> pd.DataFrame:
    """
    基于 sklearn 的 KNN 填补（Euclidean）
    ===================================================
    使用 KNNImputer，计算样本间欧氏距离（忽略 NaN）。

    :param X: pd.DataFrame
        基因型矩阵。
    :param n_neighbors: int, default=5
        近邻数量。
    :param metric: str, default="nan_euclidean"
        距离度量方式。
    :return: pd.DataFrame
        填补后的矩阵。

    说明:
        - 使用 sklearn.impute.KNNImputer；
        - 自动忽略 NaN 计算距离；
        - 大型数据集会自动提示性能警告。
    """
    if X.shape[0] > 5000:
        print(f"[WARN] Large dataset detected ({X.shape[0]} samples) — KNN imputation may be slow.")

    imputer = KNNImputer(n_neighbors=n_neighbors, metric=metric)
    M = imputer.fit_transform(X)
    print(f"[OK] sklearn.KNNImputer complete — metric={metric}, k={n_neighbors}")
    return pd.DataFrame(M, columns=X.columns, index=X.index)


def _fill_knn_hamming_abs(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, strategy: str = "mode", random_state: int | None = None) -> pd.DataFrame:
    """
    Hamming(abs) 等权 KNN 填补
    ===================================================
    使用未归一化汉明距离的 KNN 填补方法。（性能较慢，仅推荐小样本使用）

    :param X: pd.DataFrame
        基因型矩阵（值 ∈ {0,1,NaN}）。
    :param n_neighbors: int, default=5
        近邻数量。
    :param fallback: float, default=1.0
        若所有邻居均缺失时的回退值。
    :param strategy: str, default="mode"
        填补策略："mode" / "round" / "mean" / "prob"。
    :param random_state: int | None
        随机种子，仅在 "prob" 下生效。
    :return: pd.DataFrame
        填补后的矩阵。

    """
    X = X.copy()
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    # sklearn 的 Hamming 不能处理 NaN，先替换为哑值
    X_tmp = X.fillna(-1)

    # 构建最近邻模型（Hamming 距离）
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="hamming")
    nbrs.fit(X_tmp)

    # 获取最近邻（包括自身）
    distances, indices = nbrs.kneighbors(X_tmp)
    distances *= n_features # 转换为“未归一化汉明距离”
    indices = indices[:, 1:] # 去掉自身

    # 执行填补
    for i in range(n_samples):
        for j in range(n_features):
            if pd.isna(X.iat[i, j]):
                vals = X.iloc[indices[i], j]
                mask = ~vals.isna()
                if mask.any():
                    if strategy == "mode":
                        X.iat[i, j] = vals[mask].mode().iat[0]
                    elif strategy == "round":
                        X.iat[i, j] = round(vals[mask].mean())
                    elif strategy == "prob":
                        p = vals[mask].mean()
                        X.iat[i, j] = rng.binomial(1, p)
                    elif strategy == "mean":
                        X.iat[i, j] = vals[mask].mean()
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                else:
                    X.iat[i, j] = fallback
    print(f"[OK] Hamming(abs) KNN complete — k={n_neighbors}, strategy={strategy}, fallback={fallback}, seed={random_state}")
    return X


def _fill_knn_hamming_adaptive(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, target_decay: float = 0.5, strategy: str = "mode", log_samples: int = 3, random_state: int | None = None, parallel: bool = True) -> pd.DataFrame:
    """
    Adaptive Weighted Hamming KNN（自适应加权填补）
    ===================================================
    采用自适应衰减 α 的加权 KNN 填补算法。
    支持向量化与并行，适合高维稀疏 SNP 矩阵。

    :param X: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :param n_neighbors: int, default=5
        近邻数量。
    :param fallback: float, default=1.0
        回退值。
    :param target_decay: float, default=0.5
        控制权重中位邻居衰减比例。
    :param strategy: str, default="mode"
        填补策略："mode"/"round"/"mean"/"prob"。
    :param log_samples: int, default=3
        日志样本数量（打印 α 与部分权重）。
    :param random_state: int | None
        随机种子。
    :param parallel: bool, default=True
        是否启用列级并行。
    :return: pd.DataFrame
        填补后的矩阵。

    特性说明:
        - 向量化 α 与权重计算；
        - 列级并行填补；
        - 可打印部分样本的权重分布；
        - 高精度、速度快，适合中大规模数据。
    """
    X = X.copy()
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    # === 替换 NaN, 构建 BallTree, 并计算未归一化汉明距离===
    X_tmp = X.fillna(-1).to_numpy()
    tree = BallTree(X_tmp, metric="hamming")  # type: ignore[arg-type]
    dist, ind = tree.query(X_tmp, k=n_neighbors + 1)
    dist, ind = dist[:, 1:], ind[:, 1:]  # 去掉自身
    dist *= n_features  # 未归一化汉明距离

    # === 向量化计算 α & 权重矩阵 ===
    median_d = np.median(dist, axis=1, keepdims=True)
    alpha = np.log(1 / target_decay) / np.maximum(median_d, 1e-9)
    weights = np.exp(-alpha * dist)

    # 随机打印部分样本日志
    log_idx = rng.choice(n_samples, size=min(log_samples, n_samples), replace=False)

    # for i in log_idx:
    #     print(f"[INFO] Sample #{i:<4d} | α={alpha[i,0]:.4f} | median_d={median_d[i,0]:.2f} | mean_d={dist[i].mean():.2f}")

    for i in log_idx:
        # 打印前 5 个邻居权重（保留两位小数）
        w_preview = np.round(weights[i, :5], 3)
        print(f"[INFO] Sample #{i:<4d} | α={alpha[i, 0]:.4f} | median_d={median_d[i, 0]:.2f} | "
              f"mean_d={dist[i].mean():.2f} | w[:5]={w_preview}")

    # === 定义单列填补函数 ===
    def _process_column(j: int) -> np.ndarray:
        """
        对单列执行缺失值填补逻辑（可并行）。
        ---------------------------------------------------
        :param j: int 列索引。

        :return: col: np.ndarray 该列填补后的完整列向量。

        逻辑:
            1. 找出该列中所有缺失样本；
            2. 对每个缺失样本，根据其 k 个邻居进行加权计算；
            3. 根据 strategy 选择不同的填补方式；
            4. 若邻居均缺失，则使用 fallback。
        """
        # 拷贝该列的原始数值（NaN 已替换为 -1）
        col = X_tmp[:, j].copy()

        # 找出该列缺失的行索引（原始 DataFrame 判断 NaN）
        missing_rows = np.where(np.isnan(X.iloc[:, j].to_numpy()))[0]
        if len(missing_rows) == 0:
            return col # 若该列无缺失值则直接返回

        # 遍历所有缺失行，逐样本填补
        for i in missing_rows:
            vals = X_tmp[ind[i], j]     # 当前样本 i 的近邻在第 j 列的取值
            mask = vals != -1           # 过滤掉缺失邻居
            if not np.any(mask):        # 若所有邻居都缺失
                col[i] = fallback
                continue
            v, w = vals[mask], weights[i, mask]  # 有效邻居值及对应权重
            weighted_mean = np.average(v, weights=w)

            # === 根据不同策略执行填补 ===
            if strategy == "mode":
                # 权重众数（通过加权投票）
                val0 = np.sum(w[v == 0])
                val1 = np.sum(w[v == 1])
                col[i] = 1.0 if val1 >= val0 else 0.0
            elif strategy == "round":
                # 四舍五入加权均值
                col[i] = np.rint(weighted_mean)
            elif strategy == "prob":
                # 概率采样（按加权均值作为概率）
                col[i] = rng.binomial(1, weighted_mean)
            elif strategy == "mean":
                # 连续加权均值
                col[i] = weighted_mean
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        return col

    # === 并行或串行执行 ===
    if parallel and n_features > 500:
        # 获取当前 CPU 核心数，动态设定并行任务数量
        n_jobs = multiprocessing.cpu_count()
        print(f"[INFO] Adaptive KNN parallel mode — {n_features} columns, jobs={n_jobs}")

        # 使用 Joblib 并行执行每列填补
        # delayed(_process_column)(j) 表示并行调用列级函数
        new_cols = Parallel(n_jobs=n_jobs)(
            delayed(_process_column)(j) for j in range(n_features)
        )

        # 将结果按列堆叠（列顺序与原始矩阵保持一致）
        X_filled = pd.DataFrame(np.column_stack(new_cols), columns=X.columns)
    else:
        # 串行模式：逐列依次执行填补
        # 仅在该列存在 NaN 时才调用填补函数，避免不必要的计算
        for j in range(n_features):
            if X.iloc[:, j].isna().any():
                X_tmp[:, j] = _process_column(j)

        # 将填补后的 numpy 数组重新封装为 DataFrame
        X_filled = pd.DataFrame(X_tmp, columns=X.columns)

    print(f"[OK] Adaptive Weighted KNN (fast) complete — k={n_neighbors}, strategy={strategy}, parallel={parallel}")
    return X_filled


def _fill_knn_hamming_balltree(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, strategy: str = "mode", parallel: bool = False, n_jobs: int = -1) -> pd.DataFrame:
    """
    BallTree 加速 Hamming KNN 填补
    ===================================================
    基于 BallTree 的 Hamming 距离加速实现，
    支持列向量化与多核并行。

    :param X: pd.DataFrame
        基因型矩阵（行=样本，列=SNP）。
    :param n_neighbors: int, default=5
        近邻数量。
    :param fallback: float, default=1.0
        若所有邻居均缺失时的回退值。
    :param strategy: str, default="mode"
        填补策略："mode"/"mean"/"round"。
    :param parallel: bool, default=False
        是否启用并行。
    :param n_jobs: int, default=-1
        并行任务数（默认所有 CPU 核）。
    :return: pd.DataFrame
        填补后的矩阵。

    特性说明:
        - BallTree 空间索引加速；
        - 向量化 + 列级并行；
        - 速度快、内存占用低；
        - 推荐中等规模数据使用。
    """
    X = X.copy()
    n_samples, n_features = X.shape
    X_tmp = X.fillna(-1).to_numpy()

    # 小数据集时禁用并行（避免进程创建开销）
    threshold = 2e6  # 元素数量阈值
    if parallel and (n_samples * n_features < threshold):
        print(f"[AUTO] Data too small ({n_samples}×{n_features}), disabling parallel for efficiency.")
        parallel = False

    # 构建 BallTree 并查询最近邻索引
    tree = BallTree(X_tmp, metric="hamming")  # type: ignore[arg-type]
    dist, ind = tree.query(X_tmp, k=n_neighbors + 1)
    ind = ind[:, 1:]  # 去掉自身

    def _process_column(j: int) -> np.ndarray:
        """
        对单列执行缺失值填补逻辑（可并行）。
        ---------------------------------------------------
        :param j: int 列索引。

        :return: col: np.ndarray 该列填补后的完整列向量。

        逻辑:
            1. 找出该列中所有缺失样本；
            2. 对每个缺失样本提取其邻居；
            3. 根据 strategy 选择不同的填补方式；
            4. 若邻居均缺失，则使用 fallback。
        """
        # 对每个缺失样本提取其邻居
        col_values = X_tmp[:, j].copy()
        new_col = col_values.copy()

        # 找出该列中缺失的样本索引
        missing_rows = np.where(np.isnan(X.iloc[:, j].to_numpy()))[0]
        if len(missing_rows) == 0:
            return new_col

        # 获取这些缺失行的邻居矩阵（形状 = [缺失数, k]）
        neighbor_matrix = X_tmp[ind[missing_rows], j]  # shape = (num_missing, k)

        # === “众数”策略 ===
        if strategy == "mode":
            for idx, vals in enumerate(neighbor_matrix):
                valid = vals[vals != -1] # 有效邻居
                if valid.size == 0:
                    new_col[missing_rows[idx]] = fallback
                else:
                    # 统计邻居值出现次数，取最多的作为填充值
                    uniq, counts = np.unique(valid, return_counts=True)
                    new_col[missing_rows[idx]] = uniq[np.argmax(counts)]

        # === “均值/四舍五入”策略 ===
        elif strategy in ("mean", "round"):
            # 将缺失邻居标记为 NaN，计算行均值
            means = np.where(neighbor_matrix != -1, neighbor_matrix, np.nan)
            means = np.nanmean(means, axis=1)
            means[np.isnan(means)] = fallback  # 若全缺失则填 fallback
            new_col[missing_rows] = np.rint(means) if strategy == "round" else means
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return new_col

    # === 并行或串行执行列级填补 ===
    if parallel:
        print(f"[INFO] Parallel BallTree filling — {n_features} columns, jobs={n_jobs}")

        # 使用 Joblib 并行地处理所有列
        # 每列调用一次 _process_column()，各进程独立运行
        new_cols = Parallel(n_jobs=n_jobs)(
            delayed(_process_column)(j) for j in range(n_features)
        )

        # 将所有处理好的列组合成新的 DataFrame
        X_filled = pd.DataFrame(np.column_stack(new_cols), columns=X.columns)
    else:
        # 串行执行模式（单线程）
        # 遍历每一列，仅在存在 NaN 时才调用填补函数
        for j in range(n_features):
            if X.iloc[:, j].isna().any():
                X_tmp[:, j] = _process_column(j)

        # 将填补完成的矩阵重新封装为 DataFrame
        X_filled = pd.DataFrame(X_tmp, columns=X.columns)

    print(f"[OK] Hamming(BallTree-fast) KNN complete — k={n_neighbors}, strategy={strategy}, parallel={parallel}")
    return X_filled


def _fill_knn_faiss(X: pd.DataFrame, n_neighbors: int = 5, n_components: int = 50, fallback: float = 1.0, strategy: str = "mode", random_state: int | None = 42) -> pd.DataFrame:
    """
    PCA + Faiss 近似 KNN 填补（超大规模版）（暂未验证）
    ===================================================
    适用于超大样本（>10,000），通过 PCA 降维
    后使用 Faiss 加速最近邻搜索。
    （暂未验证）

    :param X: pd.DataFrame
        基因型矩阵（值 ∈ {0,1,NaN}）。
    :param n_neighbors: int
        近邻数量。
    :param n_components: int, default=50
        PCA 降维维数。
    :param fallback: float
        若邻居均缺失时使用该值。
    :param strategy: str
        填补策略："mode"/"mean"/"round"。
    :param random_state: int | None
        随机种子。
    :return: pd.DataFrame
        填补后的矩阵。

    特性说明:
        - 支持 GPU / CPU 加速；
        - 可处理十万级样本；
        - 精度略低但速度极快。
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        raise ImportError("Faiss 未安装，请运行: pip install faiss-cpu 或 faiss-gpu")

    X = X.copy()
    X_tmp = X.fillna(X.mean())
    print(f"[INFO] PCA降维至 {n_components} 维...")
    Z = PCA(n_components=n_components, random_state=random_state).fit_transform(X_tmp)

    print("[INFO] 构建 Faiss 索引...")
    index = faiss.IndexFlatL2(Z.shape[1])
    index.add(Z.astype("float32"))  # type: ignore[attr-defined]
    D, I = index.search(Z.astype("float32"), n_neighbors + 1)  # type: ignore[attr-defined]
    I = I[:, 1:]

    for i in range(X.shape[0]):
        nan_cols = np.where(X.iloc[i].isna())[0]
        for j in nan_cols:
            vals = X.iloc[I[i], j].dropna()
            if vals.empty:
                X.iat[i, j] = fallback
                continue
            if strategy == "mode":
                X.iat[i, j] = vals.mode().iat[0]
            elif strategy == "mean":
                X.iat[i, j] = vals.mean()
            elif strategy == "round":
                X.iat[i, j] = round(vals.mean())
    print(f"[OK] PCA+Faiss KNN complete — k={n_neighbors}, dim={n_components}, strategy={strategy}")
    return X


def _fill_knn_auto(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, missing_threshold: float = 0.3, random_state: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    自动选择最优 KNN 填补算法（Auto-select）
    ===================================================
    根据样本规模与缺失率自动选择最佳策略。

    策略逻辑：
        - 高缺失率 ≥ 30%          → Adaptive Hamming (prob)
        - 小样本 (< 500)           → Hamming(abs)
        - 中等规模 (500 ≤ n ≤ 5000) → BallTree
        - 大规模 (5000 < n ≤ 10000) → Adaptive Weighted Hamming
        - 特大规模 (> 10000)        → PCA + Faiss

    :param X: pd.DataFrame
        基因型矩阵。
    :param n_neighbors: int
        近邻数量。
    :param fallback: float
        回退值。
    :param missing_threshold: float
        缺失率阈值（默认 0.3）。
    :param random_state: int | None
        随机种子。
    :param verbose: bool
        是否打印日志。

    :return: pd.DataFrame
        填补后的矩阵。
    """
    # === 基础信息 ===
    n_samples, n_features = X.shape
    missing_rate = X.isna().mean().mean()

    if verbose:
        print(f"[INFO] Data shape: {n_samples}×{n_features} | Missing rate={missing_rate:.2%}")


    # === 1. 高缺失率优先：使用自适应概率模式 ===
    if missing_rate >= missing_threshold:
        if verbose:
            print("[AUTO] High missingness detected — using Adaptive Hamming (probabilistic mode).")
        return _fill_knn_hamming_adaptive(
            X,
            n_neighbors=n_neighbors,
            fallback=fallback,
            target_decay=0.4,
            strategy="prob",
            random_state=random_state,
        )

    # === 2. 小样本 ===
    if n_samples < 500:
        if verbose:
            print("[AUTO] Small dataset detected — using Hamming(abs) (equal-weight mode).")
        return _fill_knn_hamming_abs(
            X, n_neighbors=n_neighbors, fallback=fallback, strategy="mode"
        )

    # === 3. 中等规模 ===
    elif n_samples <= 5000:
        if verbose:
            print("[AUTO] Medium dataset — using BallTree-fast Hamming KNN (parallel optimized).")
        return _fill_knn_hamming_balltree(
            X,
            n_neighbors=n_neighbors,
            fallback=fallback,
            strategy="mode",
            parallel=True,
        )

    # === 4.大规模 ===
    elif n_samples <= 10000:
        if verbose:
            print("[AUTO] Large dataset — using Adaptive Weighted Hamming KNN (weighted decay).")
        return _fill_knn_hamming_adaptive(
            X,
            n_neighbors=n_neighbors,
            fallback=fallback,
            target_decay=0.5,
            strategy="mode",
            random_state=random_state,
        )

    # === 5. 特大规模 ===
    else:
        if verbose:
            print("[AUTO] Very large dataset — trying PCA + Faiss approximate KNN.")
        try:
            return _fill_knn_faiss(
                X,
                n_neighbors=n_neighbors,
                n_components=50,
                fallback=fallback,
                strategy="mode",
                random_state=random_state,
            )
        except ImportError:
            print("[WARN] Faiss not installed — fallback to Adaptive Weighted Hamming.")
            return _fill_knn_hamming_adaptive(
                X,
                n_neighbors=n_neighbors,
                fallback=fallback,
                target_decay=0.5,
                strategy="mode",
                random_state=random_state,
            )


def impute_missing(X: pd.DataFrame, method: str = "mode", n_neighbors: int = 5) -> pd.DataFrame:
    """
    通用缺失值填补入口函数
    ===================================================
    根据参数选择填补方法。

    支持方法：
        - "mode"                  → 众数填补；
        - "mean"                  → 均值填补；
        - "knn"                   → sklearn KNN；
        - "knn_hamming_abs"       → 等权 Hamming；
        - "knn_hamming_adaptive"  → 自适应加权 Hamming；
        - "knn_hamming_balltree"  → BallTree 加速；
        - "knn_hamming_faiss"     → PCA + Faiss 近似 KNN 填补
        - "knn_auto"              → 自动选择策略。

    :param X: pd.DataFrame
        基因型矩阵（0=参考等位基因, 1=变异等位基因, 3=缺失）。
    :param method: str
        填补方法。
    :param n_neighbors: int, default=5
        KNN 近邻数。
    :return: pd.DataFrame
        填补后的矩阵。
    """
    print(f"[INFO] Impute missing values with method: {method}")
    method = method.lower()

    # 将编码3替换为NaN以便填补
    Z = X.replace(3, np.nan)
    before_nans = Z.isna().sum().sum()

    if method == "mode":
        filled =  _fill_mode(Z)
    elif method == "mean":
        filled =  _fill_mean(Z)
    elif method == "knn":
        filled =  _fill_knn(Z, n_neighbors=n_neighbors)
    elif method == "knn_hamming_abs":
        filled =  _fill_knn_hamming_abs(Z, n_neighbors=n_neighbors, random_state=42)
    elif method == "knn_hamming_adaptive":
        filled =  _fill_knn_hamming_adaptive(Z, n_neighbors=n_neighbors, random_state=42)
    elif method == "knn_hamming_balltree":
        filled = _fill_knn_hamming_balltree(Z, n_neighbors=n_neighbors,parallel=True)
    elif method == "knn_faiss":
        filled = _fill_knn_faiss(Z, n_neighbors=n_neighbors, random_state=42)
    elif method == "knn_auto":
        filled =  _fill_knn_auto(Z, n_neighbors=n_neighbors, random_state=42)
    else:
        raise ValueError(f"未知填补方法: {method}")

    after_nans = filled.isna().sum().sum()
    replaced = before_nans - after_nans

    if after_nans > 0:
        print(f"[WARN] 填补后仍存在 {after_nans} 个 NaN（可能为异常列）。")
    print(f"[OK] {method.upper():<5} 填补完成 | 替换缺失值数: {replaced} | 残留 NaN: {after_nans}")

    return filled