import pandas as pd
from ancient_dna import io


def test_load_geno_and_save(tmp_dir):
    """测试基因型矩阵加载与保存"""
    df = pd.DataFrame({"Genetic ID": ["A", "B"], "SNP1": [0, 1], "SNP2": [1, 3]})
    path = tmp_dir / "geno.csv"
    df.to_csv(path, sep=";", index=False)

    ids, X, snp_cols = io.load_geno(path)
    assert len(ids) == 2
    assert isinstance(X, pd.DataFrame)
    assert set(snp_cols) == {"SNP1", "SNP2"}

    out = tmp_dir / "geno_out.csv"
    io.save_csv(X, out)
    assert out.exists()
