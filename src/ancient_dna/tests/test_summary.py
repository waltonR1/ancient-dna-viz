import pandas as pd
from ancient_dna import summary


def test_build_missing_report():
    """测试缺失率报告"""
    s1 = pd.Series([0.1, 0.2, 0.3])
    s2 = pd.Series([0.05, 0.1, 0.15])
    df = summary.build_missing_report(s1, s2)
    assert "sample_missing_mean" in df.columns
    assert isinstance(df, pd.DataFrame)


def test_build_embedding_report(embedding_data):
    """测试嵌入统计报告"""
    rep = summary.build_embedding_report(embedding_data)
    assert {"Mean", "StdDev", "Min", "Max"}.issubset(rep.columns)