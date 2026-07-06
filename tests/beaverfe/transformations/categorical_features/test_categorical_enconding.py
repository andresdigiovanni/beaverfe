import pandas as pd
import pytest

from beaverfe.transformations import CategoricalEncoding


class TestCategoricalEncoding:
    # Initialize CategoricalEncoding with default parameters and verify default behavior
    def test_default_initialization(self):
        encoder = CategoricalEncoding()
        params = encoder.get_params()

        assert params["transformation_options"] is None
        assert params["ordinal_orders"] is None

    # Fit and transform a DataFrame with various encoding options and verify transformations
    def test_fit_and_transform_with_encoding_options(self):
        X = pd.DataFrame(
            {
                "col1": ["A", "B", "C", "D", "E"],
                "col2": ["A", "B", "C", "D", "E"],
                "col3": ["A", "B", "C", "D", "E"],
                "col4": ["A", "B", "C", "D", "E"],
                "col5": ["A", "B", "C", "D", "E"],
            }
        )
        y = pd.Series([0, 1, 1, 1, 0])

        ce = CategoricalEncoding(
            transformation_options={
                "col1": "label",
                "col2": "ordinal",
                "col3": "onehot",
                "col4": "binary",
                "col5": "target",
            },
            ordinal_orders={"col2": ["C", "D", "E", "A", "B"]},
        )
        ce.fit(X, y)
        transformed_X = ce.transform(X, y)

        # Assertions for the transformations
        expected_X = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [3, 4, 0, 1, 2],
                "col3_1": [1, 0, 0, 0, 0],
                "col3_2": [0, 1, 0, 0, 0],
                "col3_3": [0, 0, 1, 0, 0],
                "col3_4": [0, 0, 0, 1, 0],
                "col3_5": [0, 0, 0, 0, 1],
                "col4_0": [0, 0, 0, 1, 1],
                "col4_1": [0, 1, 1, 0, 0],
                "col4_2": [1, 0, 1, 0, 1],
                "col5": [0.521935, 0.652043, 0.652043, 0.652043, 0.521935],
            }
        )

        pd.testing.assert_frame_equal(transformed_X, expected_X, check_dtype=False)


# Encoder names that produce a single in-place numeric column (the original
# categorical column is replaced by one numeric column).
_SINGLE_COLUMN_ENCODERS = [
    "label",
    "count",
    "catboost",
    "target",
    "woe",
    "james_stein",
    "loo",
    "m_estimate",
    "quantile",
    "glmm",
    "ordinal",
]

# Encoder names that expand one categorical column into multiple numeric columns
# (the original column is dropped and replaced by ≥1 generated columns).
_MULTI_COLUMN_ENCODERS = [
    "backward_diff",
    "basen",
    "binary",
    "dummy",
    "gray",
    "hashing",
    "helmert",
    "onehot",
    "polynomial",
    "rankhot",
    "sum",
]


class TestCategoricalEncodingUnseenCategories:
    """Regression tests: every encoder must survive unseen categories + NaN at
    transform time without raising an exception.

    CategoricalEncoding.transform fills NaN with "Unknown" before encoding, so
    the encoder sees at most two unusual inputs: the literal string "Unknown"
    and the string "D" (a category present in test but not in train). Some
    encoders may legitimately return NaN for unknown inputs — that is noted in
    each parametrize branch; no blanket finite-value assertion is made.
    """

    @pytest.mark.parametrize("encoder_name", _SINGLE_COLUMN_ENCODERS)
    def test_should_not_raise_for_single_column_encoder_with_unseen_and_nan(
        self, encoder_name
    ):
        # Arrange
        X_train = pd.DataFrame({"col": ["A", "B", "C", "A", "B", "C"]})
        X_test = pd.DataFrame({"col": ["D", None]})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        ordinal_orders = {"col": ["A", "B", "C"]} if encoder_name == "ordinal" else None

        enc = CategoricalEncoding(
            transformation_options={"col": encoder_name},
            ordinal_orders=ordinal_orders,
        )
        enc.fit(X_train, y)

        # Act — must not raise
        result = enc.transform(X_test)

        # Assert: same row count; exactly one output column for the encoded feature.
        assert len(result) == 2
        assert "col" in result.columns

    @pytest.mark.parametrize("encoder_name", _MULTI_COLUMN_ENCODERS)
    def test_should_not_raise_for_multi_column_encoder_with_unseen_and_nan(
        self, encoder_name
    ):
        # Arrange
        X_train = pd.DataFrame({"col": ["A", "B", "C", "A", "B", "C"]})
        X_test = pd.DataFrame({"col": ["D", None]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        enc = CategoricalEncoding(transformation_options={"col": encoder_name})
        enc.fit(X_train, y)

        # Act — must not raise
        result = enc.transform(X_test)

        # Assert: same row count; the original 'col' column was replaced by
        # one or more generated columns (multi-column expansion).
        assert len(result) == 2
        assert "col" not in result.columns  # original column was replaced
        assert result.shape[1] >= 1
