import pytest
import pandas as pd
from ancient_dna import visualize

def test_compute_embeddings_tsne(sample_data):
    """测试 t-SNE 降维"""
    emb = visualize.compute_embeddings(
        sample_data,
        method="tsne",
        n_components=2,
        perplexity=2,   # ✅ 避免默认 30 大于样本数
        random_state=42
    )
    assert emb.shape[1] == 2

def test_compute_embeddings_mds(sample_data):
    """测试 MDS 降维"""
    emb = visualize.compute_embeddings(sample_data, method="mds", n_components=2)
    assert emb.shape[1] == 2

def test_compute_embeddings_isomap(sample_data):
    """测试 Isomap 降维"""
    emb = visualize.compute_embeddings(
        sample_data,
        method="isomap",
        n_components=2,
        n_neighbors=2   # ✅ 降低邻居数，必须小于样本数
    )
    assert emb.shape[1] == 2
    assert list(emb.columns) == ["Dim1", "Dim2"]


def test_plot_embedding_show(monkeypatch):
    import matplotlib.pyplot as plt
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    visualize.plot_embedding(pd.DataFrame({"Dim1":[0,1],"Dim2":[1,0]}))
    assert called


def test_compute_embeddings(sample_data):
    """测试降维计算"""
    emb = visualize.compute_embeddings(sample_data, method="umap", n_components=2, random_state=42)
    assert emb.shape[1] == 2
    assert list(emb.columns) == ["Dim1", "Dim2"]


def test_plot_embedding(embedding_data, meta_labels, tmp_dir):
    """测试绘图输出"""
    path = tmp_dir / "plot.png"
    visualize.plot_embedding(embedding_data, labels=meta_labels, title="Test Embedding", save_path=path)
    assert path.exists()
