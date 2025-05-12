import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from beaverfe import BeaverPipeline, auto_feature_pipeline
from beaverfe.auto_parameters.shared import evaluate_model


@pytest.fixture
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def test_auto_feature_pipeline(load_data):
    df = load_data
    x, y = df.drop(columns="target"), df["target"]

    model = KNeighborsClassifier()
    scoring = "accuracy"
    direction = "maximize"

    # Baseline score (before transformations)
    x_base = x.fillna(0)
    baseline_score = evaluate_model(x_base, y, model, scoring)
    assert baseline_score > 0, "Baseline score should be a positive number."

    # Apply auto transformations
    transformations = auto_feature_pipeline(x, y, model, scoring, direction)

    # Ensure that transformations are returned as a list
    assert isinstance(
        transformations, list
    ), "Transformations should be returned as a list."
    assert (
        len(transformations) > 0
    ), "There should be at least one transformation applied."

    # Check that transformed data produces a valid model score
    transformer = BeaverPipeline(transformations)
    transformed_score = evaluate_model(x, y, model, scoring, transformer)
    assert transformed_score > 0, "Transformed score should be a positive number."

    # Check if the transformed score improved over the baseline
    assert transformed_score >= (baseline_score) - (
        baseline_score * 0.1
    ), "Transformed score should be equal or better than the baseline."
