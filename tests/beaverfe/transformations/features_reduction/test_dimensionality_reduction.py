import numpy as np
import pandas as pd
import pytest

from beaverfe.transformations import DimensionalityReduction


@pytest.fixture
def X():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {f"col{i}": rng.normal(size=30) for i in range(5)},
    )


class TestDimensionalityReductionPCA:
    def test_should_reduce_to_requested_number_of_components(self, X):
        reducer = DimensionalityReduction(
            features=list(X.columns), method="pca", n_components=2
        )

        result = reducer.fit_transform(X)

        assert list(result.columns) == ["pca_1", "pca_2"]
        assert result.shape == (30, 2)

    def test_should_drop_original_feature_columns(self, X):
        reducer = DimensionalityReduction(
            features=list(X.columns), method="pca", n_components=2
        )

        result = reducer.fit_transform(X)

        assert not set(X.columns) & set(result.columns)

    def test_should_preserve_untouched_columns(self, X):
        X_with_extra = X.copy()
        X_with_extra["untouched"] = range(30)
        reducer = DimensionalityReduction(
            features=list(X.columns), method="pca", n_components=2
        )

        result = reducer.fit_transform(X_with_extra)

        assert "untouched" in result.columns


class TestDimensionalityReductionTruncatedSVD:
    def test_should_reduce_to_requested_number_of_components(self, X):
        reducer = DimensionalityReduction(
            features=list(X.columns), method="truncated_svd", n_components=3
        )

        result = reducer.fit_transform(X)

        assert result.shape == (30, 3)


class TestDimensionalityReductionLDA:
    def test_should_require_target_for_lda(self, X):
        reducer = DimensionalityReduction(
            features=list(X.columns), method="lda", n_components=1
        )

        with pytest.raises(ValueError, match="LDA requires target values"):
            reducer.fit(X, y=None)

    def test_should_reduce_using_target_when_lda(self, X):
        y = [0, 1] * 15
        reducer = DimensionalityReduction(
            features=list(X.columns), method="lda", n_components=1
        )

        result = reducer.fit_transform(X, y)

        assert result.shape == (30, 1)


class TestDimensionalityReductionLLE:
    def test_should_not_force_dense_eigen_solver(self, X):
        """Regression test: LLE must not hardcode eigen_solver="dense"
        (an O(n^3) solve regardless of dataset size); it should use
        sklearn's own size-aware default instead."""
        reducer = DimensionalityReduction(
            features=list(X.columns), method="lle", n_components=2
        )

        reducer.fit(X)

        assert reducer._reducer.eigen_solver == "auto"


class TestDimensionalityReductionNComponentsClamping:
    def test_should_clamp_n_components_to_feature_count_for_pca(self, X):
        # n_components(10) > n_features(5) must not raise
        reducer = DimensionalityReduction(
            features=list(X.columns), method="pca", n_components=10
        )

        result = reducer.fit_transform(X)

        # Output width capped at the number of input features
        assert result.shape[1] == len(X.columns)

    def test_should_clamp_n_components_to_feature_count_for_truncated_svd(self, X):
        reducer = DimensionalityReduction(
            features=list(X.columns), method="truncated_svd", n_components=99
        )

        result = reducer.fit_transform(X)

        assert result.shape[1] == len(X.columns)

    def test_should_not_clamp_when_n_components_already_within_bounds(self, X):
        reducer = DimensionalityReduction(
            features=list(X.columns), method="pca", n_components=2
        )

        result = reducer.fit_transform(X)

        assert result.shape[1] == 2


class TestDimensionalityReductionErrors:
    def test_should_raise_for_unknown_method(self, X):
        reducer = DimensionalityReduction(
            features=list(X.columns), method="not_a_method", n_components=2
        )

        with pytest.raises(ValueError, match="Unknown reduction method"):
            reducer.fit(X)


class TestDimensionalityReductionParams:
    def test_should_round_trip_get_set_params(self):
        reducer = DimensionalityReduction(
            features=["a", "b"], method="pca", n_components=2
        )

        params = reducer.get_params()
        reducer.set_params(method="truncated_svd")

        assert params == {
            "features": ["a", "b"],
            "method": "pca",
            "n_components": 2,
            "track_columns": False,
        }
        assert reducer.method == "truncated_svd"
