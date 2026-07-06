import warnings

import numpy as np
import pandas as pd

from beaverfe.transformations import MathematicalOperations


class TestMathematicalOperations:
    # Initialize MathematicalOperations with default operations_options and transform data
    def test_default_operations_options(self):
        # Create a sample DataFrame
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Initialize with default operations_options
        transformer = MathematicalOperations()

        # Transform the data
        transformed_data = transformer.transform(data)

        # Assert that the transformed data is the same as input data
        pd.testing.assert_frame_equal(transformed_data, data)

    # Handle division by zero gracefully without raising errors, and inject a
    # flag column to preserve the structural "undefined" information.
    def test_division_by_zero_handling(self):
        # Create a sample DataFrame with a zero value
        data = pd.DataFrame({"A": [1, 2, 0], "B": [4, 0, 6]})

        # Initialize with division operation
        transformer = MathematicalOperations(operations_options=[("A", "B", "divide")])

        # Transform the data
        transformed_data = transformer.transform(data)

        # Row 1: A/B = 1/4 = 0.25, valid
        # Row 2: A/B = 2/0 = inf -> 0, invalid (B=0)
        # Row 3: A/B = 0/6 = 0.0, valid (zero numerator is not undefined)
        assert transformed_data["A__divide__B"].tolist() == [0.25, 0.0, 0.0]

        # The __is_invalid flag must exist and be 1 only for the div-by-zero row
        assert "A__divide__B__is_invalid" in transformed_data.columns
        assert transformed_data["A__divide__B__is_invalid"].tolist() == [0, 1, 0]

    def test_inf_result_generates_is_invalid_flag(self):
        # Division by zero produces inf, which must be flagged
        data = pd.DataFrame({"X": [10.0, 5.0], "Y": [0.0, 2.0]})
        transformer = MathematicalOperations(operations_options=[("X", "Y", "divide")])

        result = transformer.transform(data)

        flag_col = "X__divide__Y__is_invalid"
        assert flag_col in result.columns
        # First row: 10/0 = inf -> invalid
        assert result[flag_col].iloc[0] == 1
        # Second row: 5/2 = 2.5 -> valid
        assert result[flag_col].iloc[1] == 0

    def test_valid_operation_produces_no_is_invalid_flag(self):
        # When no inf or NaN results occur, no __is_invalid column should appear
        data = pd.DataFrame({"P": [3.0, 6.0, 9.0], "Q": [1.0, 2.0, 3.0]})
        transformer = MathematicalOperations(operations_options=[("P", "Q", "divide")])

        result = transformer.transform(data)

        assert "P__divide__Q__is_invalid" not in result.columns

    def test_should_raise_no_performance_warning_when_many_operations_applied(self):
        # Arrange: 5 numeric columns with all 10 pairwise add operations.
        cols = ["a", "b", "c", "d", "e"]
        data = pd.DataFrame({c: list(range(1, 11)) for c in cols})
        operations = [
            (c1, c2, "add") for i, c1 in enumerate(cols) for c2 in cols[i + 1 :]
        ]
        transformer = MathematicalOperations(operations_options=operations)

        # Act & Assert: no PerformanceWarning should surface.
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            transformer.transform(data)

        performance_warnings = [
            w for w in recorded if issubclass(w.category, pd.errors.PerformanceWarning)
        ]
        assert performance_warnings == []

    def test_should_flag_all_zero_column_division_results_in_all_inf(self):
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [0.0, 0.0, 0.0]})
        transformer = MathematicalOperations(operations_options=[("A", "B", "divide")])

        result = transformer.transform(data)

        assert result["A__divide__B__is_invalid"].tolist() == [1, 1, 1]
        assert result["A__divide__B"].tolist() == [0.0, 0.0, 0.0]

    def test_should_handle_preexisting_nan_in_input_column(self):
        data = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [2.0, 2.0, 2.0]})
        transformer = MathematicalOperations(operations_options=[("A", "B", "divide")])

        result = transformer.transform(data)

        # Row 1: NaN / 2.0 = NaN (pre-existing NaN, not from the operation itself),
        # still correctly flagged as invalid.
        assert result["A__divide__B__is_invalid"][1] == 1
        assert result["A__divide__B"].iloc[1] == 0.0

    def test_should_preserve_tracked_columns_across_chained_transformers(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        t1 = MathematicalOperations(
            operations_options=[("a", "b", "add")], track_columns=True
        )
        transformed = t1.transform(data)

        t2 = MathematicalOperations(
            operations_options=[("a__add__b", "a", "multiply")], track_columns=True
        )
        t2.transform(transformed)

        assert "a__add__b" in t1.tracked_columns
        assert "a__add__b__multiply__a" in t2.tracked_columns

    def test_should_respect_fit_state_when_transform_sees_different_invalid_pattern(
        self,
    ):
        """fit() is a no-op for this transformer, so a different invalid pattern
        at transform time must still be computed fresh and correctly."""
        X_fit = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0]})
        X_transform = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [0.0, 0.0, 0.0]})

        transformer = MathematicalOperations(operations_options=[("A", "B", "divide")])
        transformer.fit(X_fit)

        result = transformer.transform(X_transform)

        assert result["A__divide__B__is_invalid"].tolist() == [1, 1, 1]
