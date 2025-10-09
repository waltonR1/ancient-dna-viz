import pytest
import pandas as pd
from ancient_dna import report


def test_build_missing_report():
    """测试缺失率报告"""
    s1 = pd.Series([0.1, 0.2, 0.3])
    s2 = pd.Series([0.05, 0.1, 0.15])
    df = report.build_missing_report(s1, s2)
    assert "sample_missing_mean" in df.columns
    assert isinstance(df, pd.DataFrame)


def test_build_embedding_report(embedding_data):
    """测试嵌入统计报告"""
    rep = report.build_embedding_report(embedding_data)
    assert set(["Mean", "StdDev", "Min", "Max"]).issubset(rep.columns)


def test_combine_and_save_reports(tmp_dir):
    """测试报告合并与保存"""
    r1 = pd.DataFrame({"a": [1, 2]})
    r2 = pd.DataFrame({"b": [3, 4]})
    combined = report.combine_reports({"r1": r1, "r2": r2})
    assert any("r1_" in c for c in combined.columns)

    out = tmp_dir / "report.csv"
    report.save_report(r1, out)
    assert out.exists()
