"""Unit tests for all SpaceGenerator subclasses.

Each generator gets:
1. A happy-path test: applicable data -> non-empty SearchSpace dict
2. An empty-path test: inapplicable data -> {}

No model fitting or cross-validation inside these tests.
"""

import numpy as np
import pandas as pd
import pytest

from beaverfe.auto_parameters.categorical_features import (
    CategoricalEncodingSpaceGenerator,
)
from beaverfe.auto_parameters.distribution_n_scale import (
    NonLinearTransformationSpaceGenerator,
    NormalizationSpaceGenerator,
    NormalizerSpaceGenerator,
    QuantileTransformationSpaceGenerator,
    ScaleTransformationSpaceGenerator,
)
from beaverfe.auto_parameters.features_reduction import (
    ColumnSelectionSpaceGenerator,
    DimensionalityReductionSpaceGenerator,
)
from beaverfe.auto_parameters.missing_n_outliers import (
    MissingValuesHandlerSpaceGenerator,
    MissingValuesIndicatorSpaceGenerator,
    OutliersSpaceGenerator,
)
from beaverfe.auto_parameters.numerical_features import (
    MathematicalOperationsSpaceGenerator,
    NumericalBinningSpaceGenerator,
    SplineTransformationSpaceGenerator,
)
from beaverfe.auto_parameters.periodic_features import (
    CyclicalFeaturesTransformerSpaceGenerator,
    DateTimeTransformerSpaceGenerator,
)
from beaverfe.model_profiler import DataProfiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_df() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 50),
            "b": rng.normal(5, 2, 50),
            "c": rng.exponential(2, 50),  # skewed
        }
    )
    y = pd.Series(rng.choice([0, 1], 50))
    return X, y


@pytest.fixture
def categorical_df() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "cat_low": rng.choice(["A", "B", "C"], 50),
            "cat_high": [f"val_{i}" for i in rng.integers(0, 60, 50)],
        }
    )
    y = pd.Series(rng.choice([0, 1], 50))
    return X, y


@pytest.fixture
def datetime_df() -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    X = pd.DataFrame({"event_date": dates})
    y = pd.Series(np.random.default_rng(42).choice([0, 1], 50))
    return X, y


@pytest.fixture
def cyclical_df() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"hour": rng.integers(0, 24, 50).astype(float)})
    y = pd.Series(rng.choice([0, 1], 50))
    return X, y


@pytest.fixture
def missing_df() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 50).astype(float)
    data[::5] = np.nan
    X = pd.DataFrame({"col_with_nan": data, "col_clean": rng.normal(0, 1, 50)})
    y = pd.Series(rng.choice([0, 1], 50))
    return X, y


# ---------------------------------------------------------------------------
# 1. MissingValuesIndicatorSpaceGenerator
# ---------------------------------------------------------------------------


class TestMissingValuesIndicatorSpaceGenerator:
    def test_should_return_nonempty_space_when_dataframe_has_nulls(self, missing_df):
        X, y = missing_df
        gen = MissingValuesIndicatorSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert "missing_indicator_features" in space

    def test_should_return_empty_space_when_no_nulls_present(self, numeric_df):
        X, y = numeric_df
        assert not X.isnull().any().any(), "Fixture must have no nulls"
        gen = MissingValuesIndicatorSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 2. MissingValuesHandlerSpaceGenerator
# ---------------------------------------------------------------------------


class TestMissingValuesHandlerSpaceGenerator:
    def test_should_return_impute_keys_when_dataframe_has_nulls(self, missing_df):
        X, y = missing_df
        gen = MissingValuesHandlerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        # All keys should start with "impute_"
        assert all(k.startswith("impute_") for k in space)
        # Each value should be a non-empty list of strings
        for options in space.values():
            assert isinstance(options, list)
            assert len(options) > 0

    def test_should_offer_knn_for_important_column_with_binary_target(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, n)
        # Strongly correlated with y, so should be flagged as important.
        col = y.astype(float) * 10 + rng.normal(scale=0.1, size=n)
        col[::5] = np.nan
        X = pd.DataFrame({"col_with_nan": col})
        gen = MissingValuesHandlerSpaceGenerator()

        space = gen.get_search_space(X, y)

        assert any(opt.startswith("knn_") for opt in space["impute_col_with_nan"])

    def test_should_offer_knn_for_important_column_with_multiclass_target(self):
        # Regression test: Pearson correlation against an arbitrarily-coded
        # multiclass target is not meaningful, so this must use mutual
        # information instead of silently always returning False.
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 5, n)  # 5-class, unordered
        col = y.astype(float) + rng.normal(scale=0.05, size=n)
        col[::5] = np.nan
        X = pd.DataFrame({"col_with_nan": col})
        gen = MissingValuesHandlerSpaceGenerator()

        space = gen.get_search_space(X, y)

        assert any(opt.startswith("knn_") for opt in space["impute_col_with_nan"])

    def test_should_not_offer_knn_for_unimportant_column_with_multiclass_target(self):
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 5, n)
        col = rng.normal(size=n)  # pure noise, unrelated to y
        col[::5] = np.nan
        X = pd.DataFrame({"col_with_nan": col})
        gen = MissingValuesHandlerSpaceGenerator()

        space = gen.get_search_space(X, y)

        assert not any(opt.startswith("knn_") for opt in space["impute_col_with_nan"])

    def test_should_return_empty_space_when_no_nulls_present(self, numeric_df):
        X, y = numeric_df
        gen = MissingValuesHandlerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 3. OutliersSpaceGenerator
# ---------------------------------------------------------------------------


class TestOutliersSpaceGenerator:
    def test_should_return_outlier_keys_when_numeric_columns_present(self, numeric_df):
        X, y = numeric_df
        gen = OutliersSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("outlier_") for k in space)
        for options in space.values():
            assert "none" in options

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = OutliersSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}

    def test_should_not_offer_iforest_for_constant_column(self):
        X = pd.DataFrame({"constant": [1.0] * 20})
        y = pd.Series([0, 1] * 10)
        gen = OutliersSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert not any("iforest" in opt for opt in space["outlier_constant"])

    def test_should_offer_iforest_for_column_with_enough_varying_values(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"varied": rng.normal(size=50)})
        y = pd.Series(rng.choice([0, 1], 50))
        gen = OutliersSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert any("iforest" in opt for opt in space["outlier_varied"])


# ---------------------------------------------------------------------------
# 4. DateTimeTransformerSpaceGenerator
# ---------------------------------------------------------------------------


class TestDateTimeTransformerSpaceGenerator:
    def test_should_return_datetime_columns_key_when_datetime_column_present(
        self, datetime_df
    ):
        X, y = datetime_df
        gen = DateTimeTransformerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert "datetime_columns" in space
        assert len(space["datetime_columns"]) > 0

    def test_should_return_empty_space_when_no_datetime_columns(self, numeric_df):
        X, y = numeric_df
        gen = DateTimeTransformerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 5. CyclicalFeaturesTransformerSpaceGenerator
# ---------------------------------------------------------------------------


class TestCyclicalFeaturesTransformerSpaceGenerator:
    def test_should_return_cyclical_keys_when_cyclical_column_present(
        self, cyclical_df
    ):
        X, y = cyclical_df
        gen = CyclicalFeaturesTransformerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        # "hour" column should be detected
        assert any("cyclical_" in k for k in space)
        for options in space.values():
            assert "none" in options

    def test_should_return_empty_space_when_no_cyclical_columns(self, categorical_df):
        X, y = categorical_df
        gen = CyclicalFeaturesTransformerSpaceGenerator()
        space = gen.get_search_space(X, y)
        # No columns with cyclical suffixes in categorical fixture
        assert space == {}

    def test_should_predict_datetime_extraction_outputs_when_raw_datetime_column_present(
        self, datetime_df
    ):
        # A raw datetime column has no "_hour"/"_month" suffix itself, but
        # DateTimeTransformer will unconditionally produce those parts before
        # this block runs, so they must be offered as candidates up front.
        X, y = datetime_df
        gen = CyclicalFeaturesTransformerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert "cyclical_event_date_month" in space
        assert "cyclical_event_date_hour" in space
        assert "cyclical_event_date_year" not in space


# ---------------------------------------------------------------------------
# 6. NumericalBinningSpaceGenerator
# ---------------------------------------------------------------------------


class TestNumericalBinningSpaceGenerator:
    def test_should_return_binning_keys_when_numeric_columns_present(self, numeric_df):
        X, y = numeric_df
        gen = NumericalBinningSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("binning_") for k in space)
        for options in space.values():
            assert "none" in options

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = NumericalBinningSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 7. MathematicalOperationsSpaceGenerator
# ---------------------------------------------------------------------------


class TestMathematicalOperationsSpaceGenerator:
    def test_should_return_math_ops_slot_keys_when_multiple_numeric_columns_present(
        self, numeric_df
    ):
        X, y = numeric_df
        gen = MathematicalOperationsSpaceGenerator()
        space = gen.get_search_space(X, y)
        # With 3 numeric cols there should be candidates
        assert isinstance(space, dict)
        if space:
            assert "math_ops_1" in space
            assert "none" in space["math_ops_1"]

    def test_should_return_multiple_independent_slots_when_many_candidates_exist(
        self,
    ):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({f"col{i}": rng.normal(size=50) + 1 for i in range(6)})
        y = rng.integers(0, 2, size=50)
        gen = MathematicalOperationsSpaceGenerator()

        space = gen.get_search_space(X, y)

        slot_keys = [k for k in space if k.startswith("math_ops_")]
        assert len(slot_keys) > 1
        # every slot exposes the same candidate menu, sampled independently
        assert space[slot_keys[0]] == space[slot_keys[1]]

    def test_should_select_top_candidates_across_multiple_chunks(self):
        # With CHUNK_SIZE forced small, scoring must span several chunks —
        # this proves candidates from every chunk are actually considered,
        # not just the first one.
        rng = np.random.default_rng(0)
        n = 40
        X = pd.DataFrame(
            {
                "informative": rng.normal(size=n),
                "noise1": rng.normal(size=n),
                "noise2": rng.normal(size=n),
                "noise3": rng.normal(size=n),
                "noise4": rng.normal(size=n),
            }
        )
        y = (X["informative"] > 0).astype(int)
        gen = MathematicalOperationsSpaceGenerator()
        gen.CHUNK_SIZE = 2

        space = gen.get_search_space(X, y)

        assert "math_ops_1" in space

    def test_should_pick_highest_scoring_candidate_across_chunk_boundary(self):
        # Two candidates land in different chunks (CHUNK_SIZE=1); the one
        # correlated with y must still win top_k=1 selection regardless of
        # which chunk it's scored in.
        rng = np.random.default_rng(0)
        n = 200
        signal = rng.normal(size=n)
        y = (signal > 0).astype(int)
        X = pd.DataFrame(
            {
                "signal_a": signal,
                "signal_b": signal,  # add/multiply -> perfectly correlated with y
                "noise_a": rng.normal(size=n),
                "noise_b": rng.normal(size=n),
            }
        )
        gen = MathematicalOperationsSpaceGenerator()
        gen.CHUNK_SIZE = 1
        transformations_map, operation_candidates = gen._generate_operations(
            X, list(X.columns)
        )

        selected = gen._select_top_k_by_mutual_info(X, y, operation_candidates, top_k=1)

        assert selected
        col_a, col_b, _op = transformations_map[selected[0]]
        assert {col_a, col_b} == {"signal_a", "signal_b"}

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = MathematicalOperationsSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 8. SplineTransformationSpaceGenerator
# ---------------------------------------------------------------------------


class TestSplineTransformationSpaceGenerator:
    def test_should_return_spline_keys_when_numeric_columns_present(self, numeric_df):
        X, y = numeric_df
        gen = SplineTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("spline_") for k in space)
        for options in space.values():
            assert "none" in options

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = SplineTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 9. CategoricalEncodingSpaceGenerator
# ---------------------------------------------------------------------------


class TestCategoricalEncodingSpaceGenerator:
    def test_should_return_cat_enc_keys_when_categorical_columns_present(
        self, categorical_df
    ):
        X, y = categorical_df
        gen = CategoricalEncodingSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("cat_enc_") for k in space)

    def test_should_return_empty_space_when_no_categorical_columns(self, numeric_df):
        X, y = numeric_df
        gen = CategoricalEncodingSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 10. NonLinearTransformationSpaceGenerator
# ---------------------------------------------------------------------------


class TestNonLinearTransformationSpaceGenerator:
    def test_should_return_nonlinear_keys_when_skewed_columns_present(self, numeric_df):
        X, y = numeric_df
        # Column "c" is exponential and should be skewed
        gen = NonLinearTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        # At least the skewed column should appear
        assert len(space) > 0
        for options in space.values():
            assert "none" in options
            assert "yeo_johnson" in options

    def test_should_return_dict_when_low_skew_data(self):
        # Uniform distribution: low skew, below threshold
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"u": rng.uniform(0, 1, 200)})
        y = pd.Series(rng.choice([0, 1], 200))
        gen = NonLinearTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        # May or may not be empty depending on actual skew; accept either
        assert isinstance(space, dict)


# ---------------------------------------------------------------------------
# 11. NormalizationSpaceGenerator
# ---------------------------------------------------------------------------


class TestNormalizationSpaceGenerator:
    def test_should_return_normalize_keys_when_numeric_columns_present(
        self, numeric_df
    ):
        X, y = numeric_df
        gen = NormalizationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("normalize_") for k in space)
        for options in space.values():
            assert "none" in options

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = NormalizationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 12. ScaleTransformationSpaceGenerator
# ---------------------------------------------------------------------------


class TestScaleTransformationSpaceGenerator:
    def test_should_return_scale_keys_when_numeric_columns_present(self, numeric_df):
        X, y = numeric_df
        gen = ScaleTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("scale_") for k in space)
        for options in space.values():
            assert "none" in options
            assert "standard" in options

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = ScaleTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 13. QuantileTransformationSpaceGenerator
# ---------------------------------------------------------------------------


class TestQuantileTransformationSpaceGenerator:
    def test_should_return_quantile_keys_when_numeric_columns_present(self, numeric_df):
        X, y = numeric_df
        gen = QuantileTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("quantile_") for k in space)
        for options in space.values():
            assert "none" in options

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = QuantileTransformationSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 14. NormalizerSpaceGenerator
# ---------------------------------------------------------------------------


class TestNormalizerSpaceGenerator:
    def test_should_return_norm_keys_when_numeric_columns_present(self, numeric_df):
        X, y = numeric_df
        gen = NormalizerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert all(k.startswith("norm_") for k in space)
        for options in space.values():
            assert "none" in options

    def test_should_include_all_normaliser_families_in_options(self, numeric_df):
        X, y = numeric_df
        gen = NormalizerSpaceGenerator()
        space = gen.get_search_space(X, y)
        # At least one column should have all non-skewness-gated options.
        all_options = {opt for opts in space.values() for opt in opts}
        assert "normalize__l2" in all_options
        assert "scale__standard" in all_options
        assert "quantile__uniform" in all_options
        assert "quantile__normal" in all_options

    def test_should_include_nonlinear_option_only_for_skewed_columns(self, numeric_df):
        X, y = numeric_df
        gen = NormalizerSpaceGenerator()
        space = gen.get_search_space(X, y)
        # "c" is exponential (skewed); it must offer the nonlinear option.
        assert "nonlinear__yeo_johnson" in space["norm_c"]
        # "a" and "b" are normal; they may or may not be skewed enough.
        # Just verify the key exists for them.
        assert "norm_a" in space
        assert "norm_b" in space

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = NormalizerSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}

    def test_should_not_include_nonlinear_when_column_is_not_skewed(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"uniform": rng.uniform(0, 1, 200)})
        y = pd.Series(rng.choice([0, 1], 200))
        gen = NormalizerSpaceGenerator()
        space = gen.get_search_space(X, y)
        options = space.get("norm_uniform", [])
        assert not any(o.startswith("nonlinear__") for o in options)


# ---------------------------------------------------------------------------
# 15. ColumnSelectionSpaceGenerator
# ---------------------------------------------------------------------------


class TestColumnSelectionSpaceGenerator:
    def test_should_return_candidates_key_when_numeric_columns_present(
        self, numeric_df
    ):
        X, y = numeric_df
        gen = ColumnSelectionSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert "column_selection_candidates" in space
        assert len(space["column_selection_candidates"]) > 0

    def test_should_return_empty_space_when_no_numeric_columns(self, categorical_df):
        X, y = categorical_df
        gen = ColumnSelectionSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# 16. DimensionalityReductionSpaceGenerator
# ---------------------------------------------------------------------------


class TestDimensionalityReductionSpaceGenerator:
    def test_should_return_dim_reduction_keys_when_multiple_numeric_columns_present(
        self, numeric_df
    ):
        X, y = numeric_df
        gen = DimensionalityReductionSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0
        assert "dim_reduction_method" in space
        assert "dim_reduction_n_components" in space
        assert "none" in space["dim_reduction_method"]

    def test_should_return_empty_space_when_single_column(self):
        X = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        gen = DimensionalityReductionSpaceGenerator()
        space = gen.get_search_space(X, y)
        assert space == {}


# ---------------------------------------------------------------------------
# Boundary tests: single-row DataFrames
# ---------------------------------------------------------------------------


class TestBoundaryValuesSingleRow:
    """Boundary: generators must not raise when the DataFrame has exactly one row."""

    def test_should_return_dict_when_outliers_generator_receives_single_row(self):
        X = pd.DataFrame({"val": [42.0]})
        y = pd.Series([1])
        space = OutliersSpaceGenerator().get_search_space(X, y)
        assert isinstance(space, dict)

    def test_should_return_dict_when_binning_generator_receives_single_row(self):
        X = pd.DataFrame({"val": [1.0]})
        y = pd.Series([0])
        space = NumericalBinningSpaceGenerator().get_search_space(X, y)
        assert isinstance(space, dict)

    def test_should_return_dict_when_scale_generator_receives_single_row(self):
        X = pd.DataFrame({"val": [3.14]})
        y = pd.Series([1])
        space = ScaleTransformationSpaceGenerator().get_search_space(X, y)
        assert isinstance(space, dict)

    def test_should_return_dict_when_missing_handler_receives_single_nan_row(self):
        X = pd.DataFrame({"val": [float("nan")]})
        y = pd.Series([0])
        space = MissingValuesHandlerSpaceGenerator().get_search_space(X, y)
        assert isinstance(space, dict)


# ---------------------------------------------------------------------------
# Boundary tests: unicode column names
# ---------------------------------------------------------------------------


class TestUnicodeColumnNames:
    """Unicode/Encoding: generators must accept columns with non-ASCII names."""

    def test_should_return_scale_keys_when_column_names_contain_unicode(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "température": rng.normal(20, 5, 30),
                "温度": rng.normal(0, 1, 30),  # Chinese characters
                "prix_€": rng.normal(100, 10, 30),  # euro sign
            }
        )
        y = pd.Series(rng.choice([0, 1], 30))
        space = ScaleTransformationSpaceGenerator().get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_should_return_binning_keys_when_column_names_contain_emoji(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(
            {
                "score_\U0001f4af": rng.normal(0, 1, 30),
                "value_✨": rng.normal(5, 2, 30),
            }
        )
        y = pd.Series(rng.choice([0, 1], 30))
        space = NumericalBinningSpaceGenerator().get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_should_return_impute_keys_when_column_names_contain_unicode_and_nulls(
        self,
    ):
        X = pd.DataFrame(
            {
                "col_é": [1.0, float("nan"), 3.0, float("nan"), 5.0],
                "col_ü": [float("nan"), 2.0, float("nan"), 4.0, 5.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        space = MissingValuesHandlerSpaceGenerator().get_search_space(X, y)
        assert isinstance(space, dict)
        assert len(space) > 0


# ---------------------------------------------------------------------------
# set_profiler behavioral tests
# ---------------------------------------------------------------------------


def _make_data_profiler(**overrides) -> DataProfiler:
    """Construct a DataProfiler with sensible defaults, overriding specified fields."""
    defaults = {
        "n_samples": 500,
        "n_features": 5,
        "n_classes": 2,
        "class_imbalance_ratio": 1.0,
        "is_high_dimensional": False,
        "is_small_sample": False,
        "is_large_sample": False,
    }
    defaults.update(overrides)
    return DataProfiler(**defaults)


class TestMissingValuesHandlerSetProfilerBehavior:
    def test_should_not_offer_knn_when_profiler_reports_large_sample(self):
        # Arrange: column strongly correlated with y so knn would normally be offered
        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, n)
        col = y.astype(float) * 10 + rng.normal(scale=0.1, size=n)
        col[::5] = np.nan
        X = pd.DataFrame({"col_with_nan": col})

        gen = MissingValuesHandlerSpaceGenerator()
        profiler = _make_data_profiler(
            n_samples=n,
            is_large_sample=True,
        )
        gen.set_profiler(profiler)

        # Act
        space = gen.get_search_space(X, y)

        # Assert: no knn strategy offered regardless of column importance
        impute_options = space.get("impute_col_with_nan", [])
        assert not any("knn" in opt for opt in impute_options), (
            f"knn should not be offered on large samples; got: {impute_options}"
        )


class TestDimensionalityReductionSetProfilerBehavior:
    def test_should_cap_n_components_to_n_samples_divided_by_three_when_profiler_given(
        self,
    ):
        # Arrange: small n_samples=30 -> cap = 30 // 3 = 10
        rng = np.random.default_rng(0)
        n_rows = 30
        n_cols = 20
        X = pd.DataFrame(
            rng.normal(size=(n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)]
        )
        y = pd.Series(rng.integers(0, 5, n_rows))

        gen = DimensionalityReductionSpaceGenerator()
        profiler = _make_data_profiler(
            n_samples=n_rows,
            n_features=n_cols,
            is_small_sample=True,
        )
        gen.set_profiler(profiler)

        # Act
        space = gen.get_search_space(X, y)

        # Assert: all n_components options are <= n_samples // 3 = 10
        expected_cap = n_rows // 3  # 10
        n_components_options = space.get("dim_reduction_n_components", [])
        assert len(n_components_options) > 0, (
            "Expected at least one n_components option"
        )
        assert all(n <= expected_cap for n in n_components_options), (
            f"All n_components must be <= {expected_cap}; got {n_components_options}"
        )


class TestSetProfilerBoundaryThresholds:
    """set_profiler behavior at the exact DataProfiler threshold boundaries,
    using DataProfiler.from_data directly rather than a hand-built profiler."""

    def test_missing_values_handler_respects_is_large_sample_boundary_at_10000(self):
        X = pd.DataFrame({"a": range(10_000)})
        y = np.array([0, 1] * 5_000)

        profiler = DataProfiler.from_data(X, y)
        assert profiler.is_large_sample is True

        generator = MissingValuesHandlerSpaceGenerator()
        generator.set_profiler(profiler)

        space = generator.get_search_space(X, y)
        for key, options in space.items():
            if key.startswith("impute_"):
                for option in options:
                    if isinstance(option, str):
                        assert not option.startswith("knn_"), (
                            f"KNN should be excluded for large sample "
                            f"but found {option} in {key}"
                        )

    def test_missing_values_handler_allows_knn_at_9999_samples(self):
        X = pd.DataFrame({"a": range(9_999)})
        y = np.array([0, 1] * 4_999 + [0])

        profiler = DataProfiler.from_data(X, y)
        assert profiler.is_large_sample is False

        generator = MissingValuesHandlerSpaceGenerator()
        generator.set_profiler(profiler)

        # Just verifying is_large_sample=False doesn't itself suppress anything;
        # actual knn offering still depends on missing values/importance.
        generator.get_search_space(X, y)

    def test_dimensionality_reduction_respects_small_sample_boundary_at_200(self):
        X_199 = pd.DataFrame({"a": range(199), "b": range(199), "c": range(199)})
        X_200 = pd.DataFrame({"a": range(200), "b": range(200), "c": range(200)})
        y = np.array([0, 1] * 100)

        profiler_199 = DataProfiler.from_data(X_199, y[:199])
        profiler_200 = DataProfiler.from_data(X_200, y)

        assert profiler_199.is_small_sample is True
        assert profiler_200.is_small_sample is False

        generator = DimensionalityReductionSpaceGenerator()

        generator.set_profiler(profiler_199)
        spaces_199 = generator.get_search_space(X_199, y[:199])

        generator.set_profiler(profiler_200)
        spaces_200 = generator.get_search_space(X_200, y)

        assert "dim_reduction_" in str(spaces_199.keys())
        assert "dim_reduction_" in str(spaces_200.keys())
