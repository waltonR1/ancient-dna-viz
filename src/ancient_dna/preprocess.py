import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors

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
    列众数填补（默认方法）。

    :param Z: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :param fallback: 当整列均为空时的回退值（默认填 1）。
    :return: 填补后的矩阵 (pd.DataFrame)。
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

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP。
    :return: 填补后的矩阵。
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
    KNN 填补（基于样本相似度），使用 sklearn.impute.KNNImputer。

    :param X: 基因型矩阵 (pd.DataFrame)。
    :param n_neighbors: 近邻数（默认 5）。
    :param metric: 距离度量方式（默认 "nan_euclidean"）。
    :return: 填补后的矩阵。
    说明:
        - 使用 sklearn.impute.KNNImputer；
        - metric='nan_euclidean' 表示计算距离时自动忽略 NaN；
        - 每个样本缺失值由最相似样本的均值替代。
    """
    if X.shape[0] > 5000:
        print(f"[WARN] Large dataset detected ({X.shape[0]} samples) — KNN imputation may be slow.")

    imputer = KNNImputer(n_neighbors=n_neighbors, metric=metric)
    M = imputer.fit_transform(X)
    return pd.DataFrame(M, columns=X.columns, index=X.index)






def _fill_knn_hamming_abs(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0) -> pd.DataFrame:
    """ 基于 sklearn.NearestNeighbors 的高性能“未归一化汉明距离” KNN 填补。 高维 SNP 数据推荐使用该版本。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP（值为 0/1/2/NaN）。
    :param n_neighbors: 近邻数量（默认 5）。 :param fallback: 若所有邻居该列均缺失时的回退值（默认 1.0）。
    :return: 填补后的矩阵 (pd.DataFrame)，与输入结构相同。
    说明： - 使用 sklearn.neighbors.NearestNeighbors(metric="hamming")；
    - 由于 sklearn.hamming 是归一化的（比例差异），结果等价于老师的未归一化版本的缩放；
    - 速度远超纯 Python 双循环实现；
    - 每个缺失值用 K 个最近邻在该列的均值填补；
    - 若无有效邻居，则用 fallback。
    """
    X = X.copy()
    n_samples, n_features = X.shape

    # sklearn 的 Hamming 不能处理 NaN，先替换为哑值（例如 -1）
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
                vals = X.iloc[indices[i], j].dropna()
                if len(vals) > 0:
                    X.iat[i, j] = vals.mean()
                else:
                    X.iat[i, j] = fallback
    return X

def _fill_knn_hamming_weighted(
    X: pd.DataFrame,
    n_neighbors: int = 5,
    fallback: float = 1.0,
    alpha: float = 1.0,
    log_samples: int = 3
) -> pd.DataFrame:
    """
    基于加权 KNN（汉明距离）的智能缺失填补 + 日志输出版。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP（值为 0/1/2/NaN）。
    :param n_neighbors: 近邻数量（默认 5）。
    :param fallback: 若所有邻居该列均缺失时的回退值（默认 1.0）。
    :param alpha: 距离衰减系数（越大则远邻惩罚越强，推荐 0.5~1.0）。
    :param log_samples: 打印日志的样本数量（默认随机展示 3 个）。
    :return: 填补后的矩阵 (pd.DataFrame)，与输入结构相同。

    日志示例：
        [INFO] Sample #12 平均邻居距离: 7.6
               α=0.8 → 权重分布 [1.00, 0.45, 0.20, 0.09, 0.04]
    """
    X = X.copy()
    n_samples, n_features = X.shape

    # sklearn 的 Hamming 不能处理 NaN，先替换为哑值
    X_tmp = X.fillna(-1)

    # 构建最近邻模型
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="hamming")
    nbrs.fit(X_tmp)

    # 获取最近邻（包括自身）
    distances, indices = nbrs.kneighbors(X_tmp)
    distances *= n_features  # 转换为未归一化汉明距离
    indices, distances = indices[:, 1:], distances[:, 1:]  # 去掉自身

    # 随机挑选样本用于日志打印
    rng = np.random.default_rng(42)
    log_idx = rng.choice(n_samples, size=min(log_samples, n_samples), replace=False)

    for i in range(n_samples):
        # 打印日志（只显示部分样本）
        if i in log_idx:
            avg_d = distances[i].mean()
            w = np.exp(-alpha * distances[i])
            w = w / w.max()  # 归一化到 [1, …]
            w_preview = np.round(w[:min(5, n_neighbors)], 2).tolist()
            print(f"[INFO] Sample #{i} 平均邻居距离: {avg_d:.2f} | α={alpha:.2f} → 权重分布 {w_preview}")

        # 逐列填补缺失
        for j in range(n_features):
            if pd.isna(X.iat[i, j]):
                neighbor_idx = indices[i]
                neighbor_dist = distances[i]

                neighbor_vals = X.iloc[neighbor_idx, j]
                mask_valid = ~neighbor_vals.isna()

                if mask_valid.any():
                    vals = neighbor_vals[mask_valid].to_numpy()
                    dists = neighbor_dist[mask_valid]
                    weights = np.exp(-alpha * dists)
                    X.iat[i, j] = np.average(vals, weights=weights)
                else:
                    X.iat[i, j] = fallback

    print(f"[OK] Distance-weighted KNN imputation complete — α={alpha}, k={n_neighbors}")
    return X

def _fill_knn_hamming_adaptive(X: pd.DataFrame, n_neighbors: int = 5, fallback: float = 1.0, target_decay: float = 0.5, log_samples: int = 3) -> pd.DataFrame:
    """
    基于加权 KNN（汉明距离）的智能缺失填补。
    高维 SNP 数据推荐使用该版本。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP（值为 0/1/2/NaN）。
    :param n_neighbors: 近邻数量（默认 5）。
    :param fallback: 若所有邻居该列均缺失时的回退值（默认 1.0）。
    :param target_decay: α 自适应目标（0~1），表示中等距离的邻居权重衰减到多少（默认 0.5）。
                         例如 target_decay=0.5 → 中位距离邻居的权重为 0.5。
    :param log_samples: 打印日志的样本数量（默认随机展示 3 个）。
    :return: 填补后的矩阵 (pd.DataFrame)，与输入结构相同。

    说明：
        - 基于 sklearn.neighbors.NearestNeighbors(metric="hamming")；
        - 使用“未归一化汉明距离”以更符合基因差异解释；
        - 每个缺失值使用邻居加权平均：
              weight_i = exp(-alpha * distance_i)
          （即距离越近权重越大）
        - 若所有邻居该列均缺失，则使用 fallback；
    """
    X = X.copy()
    n_samples, n_features = X.shape

    # sklearn 的 Hamming 不能处理 NaN，先替换为哑值（例如 -1）
    X_tmp = X.fillna(-1)

    # 构建最近邻模型（Hamming 距离）
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="hamming")
    nbrs.fit(X_tmp)

    # 获取最近邻（包括自身）
    distances, indices = nbrs.kneighbors(X_tmp)
    distances *= n_features  # 转换为“未归一化汉明距离”
    indices = indices[:, 1:]     # 去掉自身
    distances = distances[:, 1:] # 同步去掉自身

    # 随机挑选样本用于日志打印
    rng = np.random.default_rng(42)
    log_idx = rng.choice(n_samples, size=min(log_samples, n_samples), replace=False)

    # 执行加权填补
    for i in range(n_samples):
        d = distances[i]
        # 自适应计算 α：保证中位距离邻居的权重衰减到 target_decay
        median_d = np.median(d)
        alpha = np.log(1 / target_decay) / median_d if median_d > 1e-9 else 1.0

        # 打印日志（只显示部分样本）
        if i in log_idx:
            avg_d = distances[i].mean()
            w = np.exp(-alpha * distances[i])
            w = w / w.max()  # 归一化到 [1, …]
            w_preview = np.round(w[:min(5, n_neighbors)], 2).tolist()
            print(f"[INFO] Sample #{i} 平均邻居距离: {avg_d:.2f} | α={alpha:.2f} → 权重分布 {w_preview}")

        for j in range(n_features):
            if pd.isna(X.iat[i, j]):
                # 当前样本的邻居及其距离
                neighbor_idx = indices[i]
                neighbor_dist = distances[i]

                # 取该列中邻居的非缺失值
                neighbor_vals = X.iloc[neighbor_idx, j]
                mask_valid = ~neighbor_vals.isna()

                if mask_valid.any():
                    vals = neighbor_vals[mask_valid].to_numpy()
                    dists = neighbor_dist[mask_valid]

                    # 计算权重（距离越大权重越小）
                    weights = np.exp(-alpha * dists)
                    X.iat[i, j] = np.average(vals, weights=weights)
                else:
                    X.iat[i, j] = fallback

    print(f"[OK] Distance-weighted KNN imputation complete — α={alpha}, k={n_neighbors}")
    return X

def _fill_knn_hybrid_hamming_euclidean(
    X: pd.DataFrame,
    n_neighbors: int = 5,
    fallback: float = 1.0,
    alpha: float = 1.0,
    log_samples: int = 3
) -> pd.DataFrame:
    """
    混合 KNN 缺失填补（Hamming + Euclidean）：
    用汉明距离确定最近邻，用欧式距离计算权重。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP（值为 0/1/2/NaN）。
    :param n_neighbors: 邻居数量（默认 5）。
    :param fallback: 若所有邻居该列均缺失时的回退值（默认 1.0）。
    :param alpha: 欧式距离的衰减系数（越大衰减越快，默认 1.0）。
    :param log_samples: 打印日志的样本数量（默认随机展示 3 个）。
    :return: 填补后的矩阵 (pd.DataFrame)。

    说明：
        - 先用 sklearn.NearestNeighbors(metric="hamming") 找出每个样本的 k 个最近邻；
        - 然后对这些邻居的特征使用欧式距离计算加权平均；
        - 每个缺失值用加权均值填补；
        - 若邻居该列全缺失，则使用 fallback。
    """
    X = X.copy()
    n_samples, n_features = X.shape

    # 替换 NaN 为占位符（Hamming 不能处理 NaN）
    X_tmp = X.fillna(-1)

    # 先用 Hamming 确定邻居
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="hamming")
    nbrs.fit(X_tmp)
    distances_ham, indices = nbrs.kneighbors(X_tmp)
    indices = indices[:, 1:]  # 去掉自身

    # 随机挑选日志样本
    rng = np.random.default_rng(42)
    log_idx = rng.choice(n_samples, size=min(log_samples, n_samples), replace=False)

    for i in range(n_samples):
        # 计算欧式距离矩阵（针对该样本与邻居）
        Xi = X_tmp.iloc[i].to_numpy()
        neigh = X_tmp.iloc[indices[i]].to_numpy()

        # 忽略 -1（缺失）位，仅计算有效位的欧式距离
        mask_valid = (Xi != -1)
        dists_eu = np.sqrt(np.nansum((neigh[:, mask_valid] - Xi[mask_valid]) ** 2, axis=1))
        dists_eu = np.where(dists_eu == 0, 1e-9, dists_eu)  # 防止除零

        # 计算权重（欧式距离越大，权重越小）
        weights = np.exp(-alpha * dists_eu)
        weights /= weights.sum()

        # 日志示例
        if i in log_idx:
            print(f"[INFO] Sample #{i} — 平均Hamming距离: {distances_ham[i,1:].mean():.3f}, "
                  f"欧式均值: {dists_eu.mean():.3f}, 权重范围: {weights.min():.3f}~{weights.max():.3f}")

        # 逐列填补
        for j in range(n_features):
            if pd.isna(X.iat[i, j]):
                neighbor_vals = X.iloc[indices[i], j]
                mask_val = ~neighbor_vals.isna()
                if mask_val.any():
                    X.iat[i, j] = np.average(neighbor_vals[mask_val], weights=weights[mask_val])
                else:
                    X.iat[i, j] = fallback

    print(f"[OK] Hybrid (Hamming→Euclidean) KNN imputation complete — k={n_neighbors}, α={alpha}")
    return X


def _fill_knn_hybrid_autoalpha(
    X: pd.DataFrame,
    n_neighbors: int = 5,
    fallback: float = 1.0,
    target_decay: float = 0.5,
    log_samples: int = 3
) -> pd.DataFrame:
    """
    混合 KNN 缺失填补（Hamming + Euclidean，自适应 α 版本）。

    特点：
        - 用 Hamming 确定邻居；
        - 用 Euclidean 计算加权；
        - 每个样本根据其邻居欧式距离分布自动确定 α；
        - 兼顾生物合理性与平滑性。

    :param X: 基因型矩阵 (pd.DataFrame)，行=样本，列=SNP（值为 0/1/2/NaN）。
    :param n_neighbors: 近邻数量（默认 5）。
    :param fallback: 若所有邻居该列均缺失时的回退值（默认 1.0）。
    :param target_decay: α 自适应目标，表示中位距离邻居的权重衰减到多少（默认 0.5）。
                         例如 target_decay=0.5 → 中位距离邻居的权重 = 0.5。
    :param log_samples: 打印日志的样本数量（默认随机展示 3 个）。
    :return: 填补后的矩阵 (pd.DataFrame)。

    说明：
        - 比固定 α 的 hybrid 更智能；
        - α 根据每个样本的欧式距离自适应调整；
        - 对高维 SNP 数据特别稳健。
    """
    X = X.copy()
    n_samples, n_features = X.shape

    # 替换 NaN 为占位符（Hamming 不能处理 NaN）
    X_tmp = X.fillna(-1)

    # Step 1️⃣: 先用 Hamming 找邻居
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="hamming")
    nbrs.fit(X_tmp)
    distances_ham, indices = nbrs.kneighbors(X_tmp)
    indices = indices[:, 1:]  # 去掉自身

    # 随机挑选样本日志
    rng = np.random.default_rng(42)
    log_idx = rng.choice(n_samples, size=min(log_samples, n_samples), replace=False)

    for i in range(n_samples):
        Xi = X_tmp.iloc[i].to_numpy()
        neigh = X_tmp.iloc[indices[i]].to_numpy()

        # 忽略 -1（缺失）位，仅计算有效位的欧式距离
        mask_valid = (Xi != -1)
        dists_eu = np.sqrt(np.nansum((neigh[:, mask_valid] - Xi[mask_valid]) ** 2, axis=1))
        dists_eu = np.where(dists_eu == 0, 1e-9, dists_eu)

        # Step 2️⃣: 自适应 α（控制中位距离邻居权重 = target_decay）
        median_d = float(np.median(dists_eu))
        alpha = np.log(1 / target_decay) / median_d if median_d > 1e-9 else 1.0

        # Step 3️⃣: 计算权重
        weights = np.exp(-alpha * dists_eu)
        weights /= weights.sum()

        # Step 4️⃣: 日志输出
        if i in log_idx:
            print(f"[INFO] Sample #{i} — 平均Hamming距: {distances_ham[i,1:].mean():.3f}, "
                  f"欧式均值: {dists_eu.mean():.3f}, α={alpha:.2f}, "
                  f"权重范围: {weights.min():.3f}~{weights.max():.3f}")

        # Step 5️⃣: 执行填补
        for j in range(n_features):
            if pd.isna(X.iat[i, j]):
                neighbor_vals = X.iloc[indices[i], j]
                mask_val = ~neighbor_vals.isna()
                if mask_val.any():
                    X.iat[i, j] = np.average(neighbor_vals[mask_val], weights=weights[mask_val])
                else:
                    X.iat[i, j] = fallback

    print(f"[OK] Hybrid Autoα KNN imputation complete — k={n_neighbors}, target_decay={target_decay}")
    return X







def impute_missing(X: pd.DataFrame, method: str = "mode", n_neighbors: int = 5) -> pd.DataFrame:
    """
    缺失值填补。

    :param X: 基因型矩阵 (pd.DataFrame)。（编码规则: 0 = 参考等位基因, 1 = 变异等位基因, 3 = 缺失）
    :param method: 填补方法（'mode' / 'mean' / 'knn' / 'knn_hamming'）。
    :param n_neighbors: KNN 填补时的近邻数（默认 5）。
    :return: 填补后的矩阵 (pd.DataFrame)。
        说明:
        - mode: 每列取众数；
        - mean: 每列取均值；
        - knn : 基于样本相似度插补；
        - 所有方法接口一致：接收 DataFrame，返回 DataFrame。
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
        filled =  _fill_knn_hamming_abs(Z, n_neighbors=n_neighbors)
    elif method == "knn_hamming_weighted":
        filled =  _fill_knn_hamming_weighted(Z, n_neighbors=n_neighbors)
    elif method == "knn_hamming_adaptive":
        filled =  _fill_knn_hamming_adaptive(Z, n_neighbors=n_neighbors)
    elif method == "knn_hybrid_hamming_euclidean":
        filled =  _fill_knn_hybrid_hamming_euclidean(Z, n_neighbors=n_neighbors)
    elif method == "knn_hybrid_autoalpha":
        filled =  _fill_knn_hybrid_autoalpha(Z, n_neighbors=n_neighbors)
    else:
        raise ValueError(f"未知填补方法: {method}")

    after_nans = filled.isna().sum().sum()
    replaced = before_nans - after_nans

    if after_nans > 0:
        print(f"[WARN] 填补后仍存在 {after_nans} 个 NaN（可能为异常列）。")
    print(f"[OK] {method.upper():<5} 填补完成 | 替换缺失值数: {replaced} | 残留 NaN: {after_nans}")

    return filled