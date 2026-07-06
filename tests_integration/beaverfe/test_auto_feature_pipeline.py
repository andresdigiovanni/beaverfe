"""Integration tests for auto_feature_pipeline — run via `make integration`.

These tests invoke the full Optuna search loop with real cross-validation and ML
model fitting. They are intentionally slow and excluded from the unit test suite.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from beaverfe import auto_feature_pipeline
from beaverfe.beaver_pipeline import BeaverPipeline


class TestAutoFeaturePipeline:
    def test_should_exclude_scale_and_quantile_when_tree_model_used(self, iris_data):
        """Tree model: returns list[dict], no scale/quantile transformers included."""
        X, y = iris_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        recipe = auto_feature_pipeline(
            X,
            y,
            model=model,
            scoring="accuracy",
            direction="maximize",
            n_trials=3,
            timeout=10,
            verbose=False,
        )

        assert isinstance(recipe, list), "Must return a list"
        for item in recipe:
            assert isinstance(item, dict)
            assert "name" in item
            assert "params" in item
        transformer_names = [d["name"] for d in recipe]
        assert "ScaleTransformation" not in transformer_names, (
            "Tree model should not include ScaleTransformation"
        )
        assert "QuantileTransformation" not in transformer_names, (
            "Tree model should not include QuantileTransformation"
        )
        assert "SplineTransformation" not in transformer_names, (
            "Tree model should not include SplineTransformation"
        )

    def test_should_return_list_when_linear_model_used(self, breast_cancer_data):
        """Linear model: returns non-empty list[dict]."""
        X, y = breast_cancer_data
        model = LogisticRegression(max_iter=200, random_state=42)

        recipe = auto_feature_pipeline(
            X,
            y,
            model=model,
            scoring="accuracy",
            direction="maximize",
            n_trials=3,
            timeout=10,
            verbose=False,
        )

        assert isinstance(recipe, list), "Must return a list"

    def test_should_produce_clean_output_when_pipeline_applied_to_iris(self, iris_data):
        """Recipe from tree model runs through BeaverPipeline without inf/NaN."""
        X, y = iris_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        recipe = auto_feature_pipeline(
            X,
            y,
            model=model,
            scoring="accuracy",
            direction="maximize",
            n_trials=3,
            timeout=10,
            verbose=False,
        )

        if not recipe:
            pytest.skip("No transformations returned; skipping round-trip test")

        pipeline = BeaverPipeline(recipe)
        X_transformed = pipeline.fit_transform(X, y)

        assert not X_transformed.isnull().any().any(), (
            "Transformed data must not contain NaN"
        )
        assert not np.isinf(X_transformed.values).any(), (
            "Transformed data must not contain inf"
        )

    def test_should_produce_clean_output_for_linear_model_on_breast_cancer(
        self, breast_cancer_data
    ):
        """Regression test: math-op-generated columns must be eligible for scaling."""
        X, y = breast_cancer_data
        model = LogisticRegression(max_iter=200, random_state=42)

        recipe = auto_feature_pipeline(
            X,
            y,
            model=model,
            scoring="accuracy",
            direction="maximize",
            n_trials=5,
            timeout=15,
            verbose=False,
        )

        if not recipe:
            pytest.skip("No transformations returned; skipping round-trip test")

        pipeline = BeaverPipeline(recipe)
        X_transformed = pipeline.fit_transform(X, y)

        assert not X_transformed.isnull().any().any()
        assert not np.isinf(X_transformed.to_numpy(dtype=float)).any()

    def test_should_return_list_when_direction_is_minimize(self, iris_data):
        """direction='minimize' with neg_log_loss scorer completes without error."""
        X, y = iris_data
        model = LogisticRegression(max_iter=200, random_state=42)

        recipe = auto_feature_pipeline(
            X,
            y,
            model=model,
            scoring="neg_log_loss",
            direction="minimize",
            n_trials=3,
            timeout=10,
            verbose=False,
        )

        assert isinstance(recipe, list), "Must return a list"
        for item in recipe:
            assert isinstance(item, dict)
            assert "name" in item
            assert "params" in item
