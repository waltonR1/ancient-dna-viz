import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    """创建一个模块级临时目录，用于保存测试输出文件"""
    d = tmp_path_factory.mktemp("tmp")
    return d


@pytest.fixture(scope="module")
def sample_data():
    """生成一个带缺失值的小型基因型矩阵"""
    X = pd.DataFrame({
        "SNP1": [0, 1, 3, 1],
        "SNP2": [1, 3, 0, np.nan],
        "SNP3": [3, 3, 1, 0]
    })
    return X


@pytest.fixture(scope="module")
def embedding_data():
    """生成一个 2D 降维结果示例"""
    return pd.DataFrame({
        "Dim1": [0.1, -0.2, 0.3],
        "Dim2": [0.4, 0.5, -0.6]
    })


@pytest.fixture(scope="module")
def meta_labels():
    """模拟类别标签"""
    return pd.Series(["A", "B", "A"])
