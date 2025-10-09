import pytest
import numpy as np
import pandas as pd
from ancient_dna import preprocess


def test_compute_missing_rates(sample_data):
    """测试缺失率计算"""
    sample_missing, snp_missing = preprocess.compute_missing_rates(sample_data)
    assert all(0 <= x <= 1 for x in sample_missing)
    assert all(0 <= x <= 1 for x in snp_missing)


def test_filter_by_missing(sample_data):
    """测试缺失率过滤函数"""
    s, c = preprocess.compute_missing_rates(sample_data)
    X_filtered = preprocess.filter_by_missing(sample_data, s, c)
    assert isinstance(X_filtered, pd.DataFrame)
    assert not X_filtered.empty


@pytest.mark.parametrize("method", ["mode", "mean", "knn"])
def test_impute_missing(sample_data, method):
    """测试三种填补方法（mode/mean/knn）"""
    X_nan = sample_data.replace(3, np.nan)
    X_filled = preprocess.impute_missing(X_nan, method=method)
    assert not X_filled.isna().any().any()
