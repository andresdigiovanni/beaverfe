import pandas as pd
import pytest
from sklearn.datasets import load_iris

from beaverfe import BeaverPipeline
from beaverfe.transformations import (
    ColumnSelection,
    MathematicalOperations,
    MissingValuesHandler,
    NonLinearTransformation,
    Normalization,
    NumericalBinning,
    OutliersHandler,
    QuantileTransformation,
    ScaleTransformation,
)


@pytest.fixture
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def test_transformations_are_applied(load_data):
    df = load_data
    x, y = df.drop(columns="target"), df["target"]

    # Define transformer with a series of transformations
    transformer = BeaverPipeline(
        [
            MissingValuesHandler(
                handling_options={
                    "sepal length (cm)": "fill_mode",
                    "sepal width (cm)": "fill_knn",
                    "petal length (cm)": "fill_mode",
                    "petal width (cm)": "interpolate",
                },
                n_neighbors={
                    "sepal width (cm)": 5,
                },
            ),
            OutliersHandler(
                handling_options={
                    "sepal length (cm)": ("median", "iqr"),
                    "sepal width (cm)": ("cap", "zscore"),
                    "petal length (cm)": ("median", "lof"),
                    "petal width (cm)": ("cap", "iforest"),
                },
                thresholds={
                    "sepal length (cm)": 1.5,
                    "sepal width (cm)": 2.5,
                },
                lof_params={
                    "petal length (cm)": {
                        "n_neighbors": 20,
                    }
                },
                iforest_params={
                    "petal width (cm)": {
                        "contamination": 0.1,
                    }
                },
            ),
            NonLinearTransformation(
                transformation_options={
                    "sepal length (cm)": "yeo_johnson",
                    "sepal length (cm)": "yeo_johnson",
                }
            ),
            QuantileTransformation(
                transformation_options={
                    "petal length (cm)": "uniform",
                    "petal width (cm)": "normal",
                }
            ),
            ScaleTransformation(
                transformation_options={
                    "sepal length (cm)": "min_max",
                    "sepal width (cm)": "min_max",
                    "petal length (cm)": "min_max",
                    "petal width (cm)": "min_max",
                }
            ),
            Normalization(
                transformation_options={
                    "sepal length (cm)": "l1",
                    "sepal width (cm)": "l2",
                }
            ),
            NumericalBinning(
                binning_options=[
                    ("sepal length (cm)", "uniform", 5),
                    ("sepal width (cm)", "quantile", 6),
                    ("petal length (cm)", "kmeans", 7),
                ]
            ),
            MathematicalOperations(
                operations_options=[
                    ("sepal length (cm)", "sepal width (cm)", "add"),
                    ("petal length (cm)", "petal width (cm)", "multiply"),
                ]
            ),
            ColumnSelection(
                columns=[
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                    "sepal length (cm)__add__sepal width (cm)",
                    "petal length (cm)__multiply__petal width (cm)",
                ]
            ),
        ]
    )

    # Fit the data
    point = int(x.shape[0] * 0.66)
    x_train, y_train = x[:point], y[:point]
    x_test, y_test = x[point:], y[point:]

    transformer.fit(x_train, y_train)
    transformed_x_test = transformer.transform(x_test)

    # Check that the transformed data contains the expected new columns
    expected_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "sepal length (cm)__add__sepal width (cm)",
        "petal length (cm)__multiply__petal width (cm)",
    ]

    assert all(
        col in transformed_x_test.columns for col in expected_columns
    ), f"Expected columns {expected_columns} not found in transformed data."

    # Ensure that transformations were applied (e.g., no NaNs after missing value handling)
    assert (
        not transformed_x_test.isna().any().any()
    ), "There should be no NaN values after transformations."

    # Validate the shape of the transformed data
    assert (
        transformed_x_test.shape[1] == len(expected_columns)
    ), f"Expected {len(expected_columns)} columns, but got {transformed_x_test.shape[1]}."
    assert (
        x_test.shape[0] == transformed_x_test.shape[0]
    ), f"Expected {x_test.shape[0]} rows, but got {transformed_x_test.shape[0]}."
