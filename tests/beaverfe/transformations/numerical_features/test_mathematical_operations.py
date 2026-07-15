import warnings

import numpy as np
import pandas as pd
import pytest

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

    def test_power_operation(self):
        data = pd.DataFrame({"A": [2.0, 3.0, 4.0], "B": [2.0, 2.0, 0.5]})
        transformer = MathematicalOperations(operations_options=[("A", "B", "power")])

        result = transformer.transform(data)

        assert result["A__power__B"].tolist() == [4.0, 9.0, 2.0]

    def test_min_operation(self):
        data = pd.DataFrame({"A": [1.0, 5.0, 3.0], "B": [4.0, 2.0, 3.0]})
        transformer = MathematicalOperations(operations_options=[("A", "B", "min")])

        result = transformer.transform(data)

        assert result["A__min__B"].tolist() == [1.0, 2.0, 3.0]

    def test_max_operation(self):
        data = pd.DataFrame({"A": [1.0, 5.0, 3.0], "B": [4.0, 2.0, 3.0]})
        transformer = MathematicalOperations(operations_options=[("A", "B", "max")])

        result = transformer.transform(data)

        assert result["A__max__B"].tolist() == [4.0, 5.0, 3.0]

    def test_log_ratio_operation(self):
        data = pd.DataFrame({"A": [np.e, 1.0], "B": [1.0, 1.0]})
        transformer = MathematicalOperations(
            operations_options=[("A", "B", "log_ratio")]
        )

        result = transformer.transform(data)

        assert result["A__log_ratio__B"].tolist() == [1.0, 0.0]

    def test_log_ratio_non_positive_generates_is_invalid_flag(self):
        # log of a non-positive ratio is NaN/-inf and must be flagged invalid.
        data = pd.DataFrame({"A": [-1.0, 2.0], "B": [1.0, 1.0]})
        transformer = MathematicalOperations(
            operations_options=[("A", "B", "log_ratio")]
        )

        result = transformer.transform(data)

        flag_col = "A__log_ratio__B__is_invalid"
        assert flag_col in result.columns

    def test_three_column_chained_operation(self):
        # (A + B) * C
        data = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0], "C": [3.0, 4.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "B", "add"), "C", "multiply")]
        )

        result = transformer.transform(data)

        new_col = "(A__add__B)__multiply__C"
        assert new_col in result.columns
        assert result[new_col].tolist() == [9.0, 20.0]

    def test_three_column_chained_operation_different_ops(self):
        # (A - B) / C
        data = pd.DataFrame({"A": [10.0, 8.0], "B": [4.0, 2.0], "C": [2.0, 3.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "B", "subtract"), "C", "divide")]
        )

        result = transformer.transform(data)

        new_col = "(A__subtract__B)__divide__C"
        assert result[new_col].tolist() == [3.0, 2.0]

    def test_three_column_operation_propagates_invalid_flag(self):
        # (A / B) where B has a zero -> inf -> flagged invalid, then * C.
        data = pd.DataFrame({"A": [1.0, 4.0], "B": [0.0, 2.0], "C": [5.0, 5.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "B", "divide"), "C", "multiply")]
        )

        result = transformer.transform(data)

        flag_col = "(A__divide__B)__multiply__C__is_invalid"
        assert flag_col in result.columns
        assert result[flag_col].tolist() == [1, 0]

    def test_three_column_operation_tracks_all_input_columns(self):
        data = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "B", "add"), "C", "multiply")],
            track_columns=True,
        )

        transformer.transform(data)

        assert transformer.tracked_columns["(A__add__B)__multiply__C"] == [
            "A",
            "B",
            "C",
        ]

    def test_unknown_operation_in_three_column_chain_is_skipped(self):
        data = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "B", "bogus_op"), "C", "multiply")]
        )

        result = transformer.transform(data)

        assert "A__bogus_op__B__multiply__C" not in result.columns

    # --- Unary operations (standalone) ---

    def test_square_unary_operation(self):
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        transformer = MathematicalOperations(operations_options=[("A", "square")])

        result = transformer.transform(data)

        assert result["A__square"].tolist() == [1.0, 4.0, 9.0]

    def test_cube_unary_operation(self):
        data = pd.DataFrame({"A": [1.0, 2.0, -3.0]})
        transformer = MathematicalOperations(operations_options=[("A", "cube")])

        result = transformer.transform(data)

        assert result["A__cube"].tolist() == [1.0, 8.0, -27.0]

    def test_sqrt_unary_operation(self):
        data = pd.DataFrame({"A": [4.0, 9.0]})
        transformer = MathematicalOperations(operations_options=[("A", "sqrt")])

        result = transformer.transform(data)

        assert result["A__sqrt"].tolist() == [2.0, 3.0]

    def test_sqrt_of_negative_generates_is_invalid_flag(self):
        data = pd.DataFrame({"A": [4.0, -9.0]})
        transformer = MathematicalOperations(operations_options=[("A", "sqrt")])

        result = transformer.transform(data)

        flag_col = "A__sqrt__is_invalid"
        assert flag_col in result.columns
        assert result[flag_col].tolist() == [0, 1]
        assert result["A__sqrt"].tolist() == [2.0, 0.0]

    def test_cbrt_unary_operation_handles_negative_values(self):
        data = pd.DataFrame({"A": [8.0, -8.0]})
        transformer = MathematicalOperations(operations_options=[("A", "cbrt")])

        result = transformer.transform(data)

        assert result["A__cbrt"].tolist() == [2.0, -2.0]
        assert "A__cbrt__is_invalid" not in result.columns

    def test_reciprocal_unary_operation(self):
        data = pd.DataFrame({"A": [2.0, 4.0]})
        transformer = MathematicalOperations(operations_options=[("A", "reciprocal")])

        result = transformer.transform(data)

        assert result["A__reciprocal"].tolist() == [0.5, 0.25]

    def test_reciprocal_of_zero_generates_is_invalid_flag(self):
        data = pd.DataFrame({"A": [2.0, 0.0]})
        transformer = MathematicalOperations(operations_options=[("A", "reciprocal")])

        result = transformer.transform(data)

        flag_col = "A__reciprocal__is_invalid"
        assert flag_col in result.columns
        assert result[flag_col].tolist() == [0, 1]

    def test_abs_unary_operation(self):
        data = pd.DataFrame({"A": [-3.0, 3.0, 0.0]})
        transformer = MathematicalOperations(operations_options=[("A", "abs")])

        result = transformer.transform(data)

        assert result["A__abs"].tolist() == [3.0, 3.0, 0.0]

    def test_unknown_unary_operation_is_skipped(self):
        data = pd.DataFrame({"A": [1.0, 2.0]})
        transformer = MathematicalOperations(operations_options=[("A", "bogus_unary")])

        result = transformer.transform(data)

        assert "A__bogus_unary" not in result.columns

    # --- Binary/triple operations with unary-transformed operands ---

    def test_binary_operation_with_unary_first_operand(self):
        # square(a) + b
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 10.0, 10.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "square"), "B", "add")]
        )

        result = transformer.transform(data)

        new_col = "(A__square)__add__B"
        assert new_col in result.columns
        assert result[new_col].tolist() == [11.0, 14.0, 19.0]

    def test_binary_operation_with_unary_second_operand(self):
        # a + sqrt(b)
        data = pd.DataFrame({"A": [1.0, 2.0], "B": [4.0, 9.0]})
        transformer = MathematicalOperations(
            operations_options=[("A", ("B", "sqrt"), "add")]
        )

        result = transformer.transform(data)

        new_col = "A__add__(B__sqrt)"
        assert new_col in result.columns
        assert result[new_col].tolist() == [3.0, 5.0]

    def test_binary_operation_with_unary_both_operands(self):
        # square(a) * cube(b)
        data = pd.DataFrame({"A": [2.0, 3.0], "B": [1.0, 2.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "square"), ("B", "cube"), "multiply")]
        )

        result = transformer.transform(data)

        new_col = "(A__square)__multiply__(B__cube)"
        assert result[new_col].tolist() == [4.0, 72.0]

    def test_triple_operation_with_unary_operands(self):
        # (a**2 + b) * sqrt(c)
        data = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [4.0, 9.0]}
        )
        transformer = MathematicalOperations(
            operations_options=[
                ((("A", "square"), "B", "add"), ("C", "sqrt"), "multiply")
            ]
        )

        result = transformer.transform(data)

        new_col = "((A__square)__add__B)__multiply__(C__sqrt)"
        assert new_col in result.columns
        # Row 1: (1 + 3) * 2 = 8 ; Row 2: (4 + 4) * 3 = 24
        assert result[new_col].tolist() == [8.0, 24.0]

    def test_triple_operation_tracks_root_columns_not_operand_tuples(self):
        data = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        transformer = MathematicalOperations(
            operations_options=[
                ((("A", "square"), "B", "add"), ("C", "sqrt"), "multiply")
            ],
            track_columns=True,
        )

        transformer.transform(data)

        new_col = "((A__square)__add__B)__multiply__(C__sqrt)"
        assert transformer.tracked_columns[new_col] == ["A", "B", "C"]

    def test_binary_operation_skipped_when_unary_operand_uses_unknown_op(self):
        data = pd.DataFrame({"A": [1.0], "B": [2.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "bogus_unary"), "B", "add")]
        )

        result = transformer.transform(data)

        assert "A__bogus_unary__add__B" not in result.columns

    # --- Recursive grammar: previously-ambiguous / previously-impossible shapes ---

    def test_nested_binary_left_associative_matches_explicit_nesting(self):
        # (A + B) * C, expressed as an explicit nested tuple instead of the
        # old ambiguous flat 5-tuple (A, "add", B, "multiply", C).
        data = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0], "C": [3.0, 4.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "B", "add"), "C", "multiply")]
        )

        result = transformer.transform(data)

        assert result["(A__add__B)__multiply__C"].tolist() == [9.0, 20.0]

    def test_nested_binary_right_associative_now_representable(self):
        # A + (B * C) — previously impossible to express at all; the old
        # flat 5-tuple grammar could only encode left-to-right chaining.
        data = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0], "C": [3.0, 4.0]})
        transformer = MathematicalOperations(
            operations_options=[("A", ("B", "C", "multiply"), "add")]
        )

        result = transformer.transform(data)

        new_col = "A__add__(B__multiply__C)"
        assert result[new_col].tolist() == [7.0, 14.0]

    def test_unary_of_composite_binary_expression(self):
        # square(A + B) — unary now wraps a composite sub-expression, not
        # just a single raw column.
        data = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]})
        transformer = MathematicalOperations(
            operations_options=[(("A", "B", "add"), "square")]
        )

        result = transformer.transform(data)

        new_col = "(A__add__B)__square"
        assert result[new_col].tolist() == [9.0, 25.0]

    def test_binary_combination_of_two_composite_expressions(self):
        # (A + B) - (C * D) — both operands of the outer binary op are
        # themselves composite expressions.
        data = pd.DataFrame(
            {"A": [5.0], "B": [3.0], "C": [2.0], "D": [4.0]}
        )
        transformer = MathematicalOperations(
            operations_options=[
                (("A", "B", "add"), ("C", "D", "multiply"), "subtract")
            ]
        )

        result = transformer.transform(data)

        new_col = "(A__add__B)__subtract__(C__multiply__D)"
        # (5 + 3) - (2 * 4) = 8 - 8 = 0
        assert result[new_col].tolist() == [0.0]

    def test_deeply_nested_expression_depth_four(self):
        # ((A**2 + B) * sqrt(C)) - D : depth-4 nested expression, proving the
        # recursion has no artificial depth cap.
        data = pd.DataFrame(
            {"A": [1.0], "B": [3.0], "C": [4.0], "D": [1.0]}
        )
        transformer = MathematicalOperations(
            operations_options=[
                (
                    ((("A", "square"), "B", "add"), ("C", "sqrt"), "multiply"),
                    "D",
                    "subtract",
                )
            ]
        )

        result = transformer.transform(data)

        new_col = "(((A__square)__add__B)__multiply__(C__sqrt))__subtract__D"
        # (1 + 3) * 2 - 1 = 8 - 1 = 7
        assert result[new_col].tolist() == [7.0]

    def test_bare_column_name_at_top_level_is_skipped(self):
        # A bare str entry at the top level applies no transformation and
        # would produce a duplicate-named column (same name as the raw
        # column) — intentionally rejected/skipped rather than silently
        # corrupting the DataFrame with a duplicate "A" column.
        data = pd.DataFrame({"A": [1.0, 2.0]})
        transformer = MathematicalOperations(operations_options=["A"])

        result = transformer.transform(data)

        assert list(result.columns) == ["A"]

    def test_five_element_flat_tuple_shape_is_no_longer_supported(self):
        # The old ambiguous flat chained shape (operand1, op1, operand2, op2,
        # operand3) is intentionally unsupported now — describe_operation
        # raises for it, so transform() skips it instead of guessing an
        # evaluation order.
        data = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        transformer = MathematicalOperations(
            operations_options=[("A", "add", "B", "multiply", "C")]
        )

        result = transformer.transform(data)

        assert list(result.columns) == ["A", "B", "C"]

        with pytest.raises(ValueError):
            MathematicalOperations.describe_operation(
                ("A", "add", "B", "multiply", "C")
            )

