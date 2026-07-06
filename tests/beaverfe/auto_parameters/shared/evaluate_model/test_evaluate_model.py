import logging

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from beaverfe.auto_parameters.shared.evaluate_model.evaluate_model import (
    CastAuxiliaryDtypes,
    build_pipeline,
    evaluate_model,
)
from beaverfe.beaver_pipeline import BeaverPipeline
from beaverfe.pipeline_blocks import CANONICAL_ORDER


class _FakeTrial:
    def __init__(self, prune_after_step=None):
        self.prune_after_step = prune_after_step
        self.reported: list[tuple[float, int]] = []

    def report(self, value, step):
        self.reported.append((value, step))

    def should_prune(self):
        return (
            self.prune_after_step is not None
            and len(self.reported) > self.prune_after_step
        )


class _IdentityTransformer(BaseEstimator, TransformerMixin):
    """Minimal pass-through transformer used as a test double."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class TestBuildPipeline:
    def test_should_have_only_transformer_and_model_when_skip_safety_net_is_true(self):
        # Arrange
        model = LogisticRegression()
        transformer = _IdentityTransformer()

        # Act
        pipeline = build_pipeline(model, transformer=transformer, skip_safety_net=True)

        # Assert: exactly [("transformer", ...), ("model", ...)] — no safety net.
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["transformer", "model"]

    def test_should_have_five_steps_when_skip_safety_net_is_false(self):
        # Arrange
        model = LogisticRegression()
        transformer = _IdentityTransformer()

        # Act
        pipeline = build_pipeline(model, transformer=transformer, skip_safety_net=False)

        # Assert: transformer + drop_datetime + cast_auxiliary_dtypes + preprocessing + model
        assert len(pipeline.steps) == 5

    def test_should_have_full_safety_net_by_default(self):
        # Arrange
        model = LogisticRegression()

        # Act
        pipeline = build_pipeline(model)

        # Assert: drop_datetime, cast_auxiliary_dtypes, preprocessing, model (no transformer)
        step_names = [name for name, _ in pipeline.steps]
        assert "drop_datetime" in step_names
        assert "preprocessing" in step_names
        assert "model" in step_names


class TestCastAuxiliaryDtypes:
    def test_should_cast_bool_column_to_numeric_when_transforming(self):
        # Arrange
        X = pd.DataFrame({"flag": [True, False, True]})
        caster = CastAuxiliaryDtypes()

        # Act
        result = caster.fit_transform(X)

        # Assert
        assert result["flag"].tolist() == [1, 0, 1]
        assert pd.api.types.is_numeric_dtype(result["flag"])

    def test_should_cast_timedelta_column_to_seconds_when_transforming(self):
        # Arrange
        X = pd.DataFrame({"duration": pd.to_timedelta([1, 2], unit="h")})
        caster = CastAuxiliaryDtypes()

        # Act
        result = caster.fit_transform(X)

        # Assert
        assert result["duration"].tolist() == [3600.0, 7200.0]

    def test_should_leave_other_columns_unchanged_when_transforming(self):
        # Arrange
        X = pd.DataFrame({"numeric": [1.0, 2.0], "category": ["a", "b"]})
        caster = CastAuxiliaryDtypes()

        # Act
        result = caster.fit_transform(X)

        # Assert
        pd.testing.assert_frame_equal(result, X)


class TestEvaluateModel:
    def test_should_not_drop_bool_and_timedelta_columns_when_scoring(self):
        # Arrange
        X = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "flag": [True, False, True, False, True, False, True, False],
                "duration": pd.to_timedelta([1, 2, 3, 4, 5, 6, 7, 8], unit="h"),
            }
        )
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        model = LogisticRegression()

        # Act
        score = evaluate_model(X, y, model, scoring="accuracy", cv=2)

        # Assert
        assert score is not None

    def test_should_report_a_score_per_fold_when_trial_is_provided(self):
        # Arrange
        X = pd.DataFrame({"numeric": list(range(20))})
        y = [0, 1] * 10
        model = LogisticRegression()
        trial = _FakeTrial()

        # Act
        score = evaluate_model(X, y, model, scoring="accuracy", cv=4, trial=trial)

        # Assert
        assert score is not None
        assert len(trial.reported) == 4
        assert [step for _, step in trial.reported] == [0, 1, 2, 3]

    def test_should_raise_trial_pruned_when_should_prune_returns_true(self):
        # Arrange
        X = pd.DataFrame({"numeric": list(range(20))})
        y = [0, 1] * 10
        model = LogisticRegression()
        trial = _FakeTrial(prune_after_step=0)

        # Act & Assert
        with pytest.raises(optuna.TrialPruned):
            evaluate_model(X, y, model, scoring="accuracy", cv=4, trial=trial)

        # Only the fold(s) up to the prune point should have run.
        assert len(trial.reported) == 1

    def test_should_complete_without_crash_when_skip_safety_net_true_on_clean_data(
        self,
    ):
        X = pd.DataFrame({"a": list(range(20)), "b": list(range(1, 21))})
        y = np.array([0, 1] * 10)
        model = LogisticRegression(max_iter=100)
        transformer = _IdentityTransformer()

        score = evaluate_model(
            X,
            y,
            model,
            scoring="accuracy",
            transformer=transformer,
            cv=3,
            skip_safety_net=True,
        )

        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_should_not_silently_produce_nan_when_skip_safety_net_true_with_rare_category(
        self,
    ):
        """Recipe includes MissingValuesHandler + CategoricalEncoding (the case
        _recipe_guarantees_clean_output signals as safe) with a rare category
        that could vanish from a CV fold — must not silently return NaN."""
        X = pd.DataFrame(
            {
                "rare_cat": ["A"] * 8 + ["B"],
                "num": list(range(9)),
            }
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
        model = LogisticRegression(max_iter=200)

        transformer_dicts = [
            {
                "name": "MissingValuesHandler",
                "params": {
                    "transformation_options": {"num": "mean"},
                    "n_neighbors": None,
                },
                "block": None,
            },
            {
                "name": "CategoricalEncoding",
                "params": {"transformation_options": {"rare_cat": "ordinal"}},
                "block": None,
            },
        ]
        pipeline = BeaverPipeline(transformer_dicts, order=CANONICAL_ORDER)

        try:
            score = evaluate_model(
                X,
                y,
                model,
                scoring="accuracy",
                transformer=pipeline,
                cv=3,
                skip_safety_net=True,
            )
            assert np.isfinite(score), "Expected finite score or exception, got NaN"
        except (ValueError, RuntimeError) as exc:
            # An early, visible exception is acceptable here — a silent NaN is not.
            logging.debug("Expected exception with rare category: %s", exc)
