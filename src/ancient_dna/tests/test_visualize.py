import pytest
import pandas as pd
from ancient_dna import visualize


def test_plot_embedding_show(monkeypatch):
    import matplotlib.pyplot as plt
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    visualize.plot_embedding(pd.DataFrame({"Dim1":[0,1],"Dim2":[1,0]}))
    assert called

def test_plot_embedding(embedding_data, meta_labels, tmp_dir):
    """测试绘图输出"""
    path = tmp_dir / "plot.png"
    visualize.plot_embedding(embedding_data, labels=meta_labels, title="Test Embedding", save_path=path)
    assert path.exists()
