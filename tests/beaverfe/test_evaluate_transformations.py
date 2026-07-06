import pandas as pd
from sklearn.linear_model import LogisticRegression

from beaverfe.evaluate_transformations import evaluate_transformations


class TestEvaluateTransformations:
    def test_should_limit_steps_when_max_steps_is_provided(self):
        # Arrange: two valid transformations, but we only want the first one.
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        model = LogisticRegression()
        transformations = [
            {
                "name": "MissingValuesHandler",
                "params": {"transformation_options": {"a": "mean"}},
            },
            {
                "name": "ScaleTransformation",
                "params": {"transformation_options": {"a": "standard"}},
            },
        ]

        # Act: only evaluate the first step
        scores = evaluate_transformations(
            transformations,
            X,
            y,
            model,
            scoring="accuracy",
            cv=2,
            plot_file=None,
            max_steps=1,
        )

        # Assert: Baseline + 1 step (not 2)
        assert len(scores) == 2
        assert scores[0]["name"] == "Baseline"
        assert scores[1]["name"] == "MissingValuesHandler"

    def test_should_reorder_transformations_to_canonical_order_before_scoring(self):
        # Arrange: deliberately listed in the wrong order. Scaling a column
        # that still has NaN would crash (StandardScaler rejects NaN), so
        # this only succeeds if BeaverPipeline reorders to CANONICAL_ORDER
        # (impute before scale) instead of using this list's order.
        X = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0]})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        model = LogisticRegression()
        transformations = [
            {
                "name": "ScaleTransformation",
                "params": {"transformation_options": {"a": "standard"}},
            },
            {
                "name": "MissingValuesHandler",
                "params": {"transformation_options": {"a": "mean"}},
            },
        ]

        # Act
        scores = evaluate_transformations(
            transformations, X, y, model, scoring="accuracy", cv=2, plot_file=None
        )

        # Assert
        assert len(scores) == 3
        assert [s["name"] for s in scores] == [
            "Baseline",
            "ScaleTransformation",
            "MissingValuesHandler",
        ]
