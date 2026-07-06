import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris


@pytest.fixture
def iris_data() -> tuple[pd.DataFrame, np.ndarray]:
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = np.array(data.target)
    return X, y


@pytest.fixture
def breast_cancer_data() -> tuple[pd.DataFrame, np.ndarray]:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = np.array(data.target)
    return X, y
