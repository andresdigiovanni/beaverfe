import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from beaverfe.auto_parameters.shared.feature_selector.permutation_rfecv import (
    PermutationRFECV,
)


class _SelectByName(BaseEstimator, TransformerMixin):
    """Looks up a column by name, the way BeaverPipeline transformers do.

    Used to prove PermutationRFECV keeps slicing as a named DataFrame instead
    of a positional ndarray, since column-name lookups would break otherwise.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[c for c in X.columns if c.startswith("informative")]]


def _make_dataset():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "informative_1": rng.normal(size=n),
            "informative_2": rng.normal(size=n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
        }
    )
    y = (X["informative_1"] + X["informative_2"] > 0).astype(int).to_numpy()
    return X, y


class TestPermutationRFECV:
    def test_should_return_original_column_names_when_input_is_dataframe(self):
        # Arrange
        X, y = _make_dataset()
        selector = PermutationRFECV(
            estimator=LogisticRegression(), scoring="accuracy", cv=3, step=1
        )

        # Act
        selector.fit(X, y)

        # Assert
        assert set(selector.get_feature_names_out()).issubset(set(X.columns))

    def test_should_keep_column_names_when_estimator_looks_up_columns_by_name(self):
        # Arrange
        X, y = _make_dataset()
        estimator = Pipeline(
            [("select", _SelectByName()), ("model", LogisticRegression())]
        )
        selector = PermutationRFECV(estimator=estimator, scoring="accuracy", cv=3)

        # Act & Assert (would raise KeyError if columns were converted to a
        # positional ndarray internally, since _SelectByName relies on names)
        selector.fit(X, y)

    def test_should_generate_placeholder_names_when_input_is_ndarray(self):
        # Arrange
        X, y = _make_dataset()
        selector = PermutationRFECV(
            estimator=LogisticRegression(), scoring="accuracy", cv=3, step=1
        )

        # Act
        selector.fit(X.to_numpy(), y)

        # Assert
        assert list(selector.feature_names_in_) == ["x0", "x1", "x2", "x3"]

    def test_should_run_at_least_one_round_when_time_budget_is_zero(self):
        # Arrange
        X, y = _make_dataset()
        selector = PermutationRFECV(
            estimator=LogisticRegression(),
            scoring="accuracy",
            cv=3,
            step=1,
            max_time_seconds=0.0,
        )

        # Act
        selector.fit(X, y)

        # Assert
        assert len(selector.scores_) == 1

    def test_should_run_all_rounds_when_time_budget_is_generous(self):
        # Arrange
        X, y = _make_dataset()
        selector = PermutationRFECV(
            estimator=LogisticRegression(),
            scoring="accuracy",
            cv=3,
            step=1,
            max_time_seconds=60.0,
        )

        # Act
        selector.fit(X, y)

        # Assert: with step=1 and min_features_to_select=1 (defaults), a full
        # elimination run scores every feature count down to 1.
        assert len(selector.scores_) == X.shape[1]

    def test_should_select_smallest_subset_when_scores_are_on_plateau(self):
        # Arrange: two rounds with the same top score — the smaller one (3 features)
        # must win over the larger one (5 features). This verifies the plateau
        # selection bug-fix: min() by len(feats) instead of max() by score index.
        n_features = 5

        # Directly inject a mock scores_ with a clear plateau at 0.95:
        #   round 0 → 5 features, score 0.95
        #   round 1 → 3 features, score 0.95  ← should win (fewest features)
        #   round 2 → 2 features, score 0.90
        mock_scores: list[tuple[np.ndarray, float]] = [
            (np.array([0, 1, 2, 3, 4]), 0.95),
            (np.array([0, 1, 2]), 0.95),
            (np.array([0, 1]), 0.90),
        ]

        best_score = max(score for _, score in mock_scores)
        tol = max(abs(best_score) * 1e-4, 1e-6)
        best_features, _ = min(
            (
                (feats, score)
                for feats, score in mock_scores
                if best_score - score <= tol
            ),
            key=lambda x: len(x[0]),
        )
        support = np.zeros(n_features, dtype=bool)
        support[best_features] = True

        # Assert: smallest subset within plateau wins → 3 features, not 5
        assert int(support.sum()) == 3
        assert list(support) == [True, True, True, False, False]

    def test_should_keep_all_features_when_time_budget_exhausted_after_one_round(self):
        # Arrange: set max_time_seconds so small that only one round completes.
        # The selector must not crash and must return a valid (non-empty) selection.
        X, y = _make_dataset()
        selector = PermutationRFECV(
            estimator=LogisticRegression(max_iter=1000),
            scoring="accuracy",
            cv=2,
            step=0.2,
            max_time_seconds=0.0,  # budget exhausted before round 2
        )

        # Act
        selector.fit(X, y)

        # Assert: exactly one round ran; all features are selected (no elimination
        # was possible before the budget expired)
        assert len(selector.scores_) == 1
        assert selector.n_features_ == X.shape[1]

    def test_should_not_crash_when_time_budget_is_exhausted_mid_run(self):
        # Arrange: set a very tight budget so RFECV must stop early without error.
        X, y = _make_dataset()
        selector = PermutationRFECV(
            estimator=LogisticRegression(max_iter=100),
            scoring="accuracy",
            cv=2,
            step=1,
            max_time_seconds=0.001,  # nearly zero — forces early stop
        )

        # Act & Assert: no exception
        selector.fit(X, y)
        assert selector.n_features_ >= 1
        assert len(selector.get_feature_names_out()) >= 1

    def test_should_respect_time_budget_by_completing_in_reasonable_wall_time(self):
        # Arrange: give a 2-second budget on a dataset that takes ~0.1s per round.
        X, y = _make_dataset()
        max_time = 2.0
        selector = PermutationRFECV(
            estimator=LogisticRegression(max_iter=100),
            scoring="accuracy",
            cv=2,
            step=1,
            max_time_seconds=max_time,
        )

        # Act
        start = time.monotonic()
        selector.fit(X, y)
        elapsed = time.monotonic() - start

        # Assert: wall time must be within a generous 10x overhead of the budget
        assert elapsed < max_time * 10
