import pandas as pd

from beaverfe.pipeline_assembler import PipelineAssembler
from beaverfe.pipeline_blocks import PipelineBlock


class _FakeTrial:
    """Deterministically returns the first option for every suggestion."""

    def __init__(self):
        self.suggested: dict[str, list] = {}

    def suggest_categorical(self, name, choices):
        self.suggested[name] = list(choices)
        return choices[0]


class TestPipelineAssemblerForcedScaling:
    def test_should_strip_none_from_norm_options_for_linear_models(self):
        # Arrange
        assembler = PipelineAssembler()
        trial = _FakeTrial()
        spaces = {"norm_a": ["none", "scale__standard", "scale__robust__25.0"]}

        # Act
        assembler.assemble(trial, spaces, model_type="linear")

        # Assert
        assert "none" not in trial.suggested["norm_a"]

    def test_should_strip_none_from_outlier_options_for_distance_based_models(self):
        # Arrange
        assembler = PipelineAssembler()
        trial = _FakeTrial()
        spaces = {"outlier_a": ["none", "cap__iqr__1.5"]}

        # Act
        assembler.assemble(trial, spaces, model_type="distance_based")

        # Assert
        assert "none" not in trial.suggested["outlier_a"]

    def test_should_keep_none_for_norm_options_for_other_models(self):
        # Arrange
        assembler = PipelineAssembler()
        spaces = {"norm_a": ["none", "scale__standard"]}

        # Act
        trial_other = _FakeTrial()
        assembler.assemble(trial_other, spaces, model_type="other")

        # Assert
        assert "none" in trial_other.suggested["norm_a"]

    def test_should_omit_key_from_default_params_when_none_is_forced_out(self):
        # Arrange
        assembler = PipelineAssembler()
        spaces = {
            "norm_a": ["none", "scale__standard"],
            "norm_b": ["none", "quantile__uniform"],
        }

        # Act: for linear models NORMALIZE is forced non-none, so norm_ keys
        # are excluded from defaults (no valid "none" option to seed).
        defaults = assembler.default_params(spaces, model_type="linear")

        # Assert: both keys absent because neither can be "none" for linear.
        assert "norm_a" not in defaults
        assert "norm_b" not in defaults


class TestPipelineAssemblerDecodeMathOps:
    def test_should_return_empty_when_all_slots_are_none(self):
        # Arrange
        assembler = PipelineAssembler()
        params = {"math_ops_1": "none", "math_ops_2": "none"}

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert result == []

    def test_should_collect_all_non_none_slots_when_multiple_selected(self):
        # Arrange
        assembler = PipelineAssembler()
        params = {
            "math_ops_1": repr(("colA", "colB", "multiply")),
            "math_ops_2": repr(("colC", "colD", "divide")),
            "math_ops_3": "none",
        }

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert len(result) == 1
        assert result[0]["name"] == "MathematicalOperations"
        assert result[0]["block"] == PipelineBlock.EXPAND_MATH
        assert set(result[0]["params"]["operations_options"]) == {
            ("colA", "colB", "multiply"),
            ("colC", "colD", "divide"),
        }

    def test_should_deduplicate_when_same_operation_picked_by_multiple_slots(self):
        # Arrange
        assembler = PipelineAssembler()
        params = {
            "math_ops_1": repr(("colA", "colB", "multiply")),
            "math_ops_2": repr(("colA", "colB", "multiply")),
        }

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert result[0]["params"]["operations_options"] == [
            ("colA", "colB", "multiply")
        ]

    def test_should_decode_three_column_chained_operation(self):
        # Arrange: (colA add colB) multiply colC
        assembler = PipelineAssembler()
        params = {
            "math_ops_1": repr((("colA", "colB", "add"), "colC", "multiply")),
        }

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert result[0]["params"]["operations_options"] == [
            (("colA", "colB", "add"), "colC", "multiply")
        ]

    def test_should_decode_binary_combination_of_two_composite_expressions(self):
        # Arrange: (colA add colB) multiply (colC subtract colD) — both
        # operands are themselves composite expressions, proving the decoder
        # is agnostic to arbitrary nesting depth/shape (it just round-trips
        # whatever ast.literal_eval parses).
        assembler = PipelineAssembler()
        expr = (("colA", "colB", "add"), ("colC", "colD", "subtract"), "multiply")
        params = {"math_ops_1": repr(expr)}

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert result[0]["params"]["operations_options"] == [expr]

    def test_should_collect_mixed_two_and_three_column_operations(self):
        # Arrange
        assembler = PipelineAssembler()
        params = {
            "math_ops_1": repr(("colA", "colB", "multiply")),
            "math_ops_2": repr((("colA", "colB", "add"), "colC", "multiply")),
        }

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert set(result[0]["params"]["operations_options"]) == {
            ("colA", "colB", "multiply"),
            (("colA", "colB", "add"), "colC", "multiply"),
        }

    def test_should_decode_standalone_unary_operation(self):
        # Arrange: standalone unary, e.g. squaring a column
        assembler = PipelineAssembler()
        params = {"math_ops_1": repr(("colA", "square"))}

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert result[0]["params"]["operations_options"] == [("colA", "square")]

    def test_should_decode_nested_unary_operand_operation(self):
        # Arrange: square(colA) + colB
        assembler = PipelineAssembler()
        params = {"math_ops_1": repr((("colA", "square"), "colB", "add"))}

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert result[0]["params"]["operations_options"] == [
            (("colA", "square"), "colB", "add")
        ]

    def test_should_skip_corrupt_or_non_tuple_values(self):
        # Arrange: invalid literal and a value that evals but isn't a tuple
        assembler = PipelineAssembler()
        params = {
            "math_ops_1": "not a valid literal(",
            "math_ops_2": "['colA', 'colB']",
        }

        # Act
        result = assembler._decode_math_ops(params)

        # Assert
        assert result == []


class TestPipelineAssemblerTreeGating:
    def test_should_exclude_spline_dicts_for_tree_models(self):
        # Arrange
        assembler = PipelineAssembler()
        params = {"spline_a": "5__3__linear"}
        spaces = {"spline_a": ["none", "5__3__linear"]}

        # Act
        result = assembler._build_pipeline_dicts(params, spaces, model_type="tree")

        # Assert
        assert result == []

    def test_should_include_spline_dicts_for_non_tree_models(self):
        # Arrange
        assembler = PipelineAssembler()
        params = {"spline_a": "5__3__linear"}
        spaces = {"spline_a": ["none", "5__3__linear"]}

        # Act
        result = assembler._build_pipeline_dicts(params, spaces, model_type="other")

        # Assert
        assert any(d["name"] == "SplineTransformation" for d in result)


class TestPipelineAssemblerDynamicSpaces:
    def test_should_discover_downstream_keys_for_math_ops_output(self):
        # Arrange
        assembler = PipelineAssembler()
        X = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "b": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0],
            }
        )
        params = {"math_ops_1": "('a', 'b', 'multiply')"}
        spaces = {"math_ops_1": ["none", "('a', 'b', 'multiply')"]}

        # Act
        dynamic_spaces = assembler._discover_dynamic_spaces(
            params, spaces, model_type="other", X=X, y=None
        )

        # Assert: generated column gets a single unified norm_ key.
        generated_col = "a__multiply__b"
        assert f"norm_{generated_col}" in dynamic_spaces
        # Outlier handling is NOT offered for math-generated columns because
        # CLEANSE_OUTLIERS runs before EXPAND_MATH in v3 canonical order.
        assert f"outlier_{generated_col}" not in dynamic_spaces

    def test_should_return_empty_when_no_expansion_block_selected(self):
        # Arrange
        assembler = PipelineAssembler()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        params = {"math_ops_1": "none"}
        spaces = {"math_ops_1": ["none", "('a', 'b', 'multiply')"]}

        # Act
        dynamic_spaces = assembler._discover_dynamic_spaces(
            params, spaces, model_type="other", X=X, y=None
        )

        # Assert
        assert dynamic_spaces == {}

    def test_should_skip_preview_entirely_for_tree_models(self):
        # Arrange
        assembler = PipelineAssembler()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        params = {"math_ops_1": "('a', 'b', 'multiply')"}
        spaces = {"math_ops_1": ["none", "('a', 'b', 'multiply')"]}

        # Act
        dynamic_spaces = assembler._discover_dynamic_spaces(
            params, spaces, model_type="tree", X=X, y=None
        )

        # Assert: every dynamic-eligible block is tree-disabled, so nothing
        # is discovered (and the preview fit never runs).
        assert dynamic_spaces == {}

    def test_should_not_suggest_dynamic_params_when_x_is_not_provided(self):
        # Arrange
        assembler = PipelineAssembler()
        trial = _FakeTrial()
        spaces = {"math_ops_1": ["none", "('a', 'b', 'multiply')"]}

        # Act
        assembler.assemble(trial, spaces, model_type="other")

        # Assert: back-compat path, no dynamic discovery attempted
        assert "math_ops_1" in trial.suggested
        assert not any(k.startswith("norm_") for k in trial.suggested)


class TestIsInvalidFlagExclusion:
    """Regression tests: __is_invalid flag columns must never enter generated_columns,
    column_selection_candidates, or the search space of any dynamic generator."""

    def test_should_exclude_is_invalid_columns_from_generated_columns_and_candidates(
        self,
    ):
        # Arrange: column 'b' has a zero so divide produces inf -> __is_invalid flag
        assembler = PipelineAssembler()
        X = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "b": [2.0, 0.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            }
        )
        params = {"math_ops_1": "('a', 'b', 'divide')"}
        spaces = {
            "math_ops_1": ["none", "('a', 'b', 'divide')"],
            "column_selection_candidates": [["a", "b"]],
        }

        # Act
        dynamic_spaces = assembler._discover_dynamic_spaces(
            params, spaces, model_type="other", X=X, y=None
        )

        # Assert: the __is_invalid flag column must not appear as a space key
        # (e.g. scale_a__divide__b__is_invalid) nor in column_selection_candidates
        for key in dynamic_spaces:
            assert "__is_invalid" not in key, (
                f"Flag column leaked into dynamic_spaces key: {key}"
            )

        candidates_list = dynamic_spaces.get("column_selection_candidates")
        if candidates_list:
            for candidate in candidates_list[0]:
                assert not candidate.endswith("__is_invalid"), (
                    f"Flag column '{candidate}' leaked into column_selection_candidates"
                )

    def test_should_include_valid_divide_result_but_not_flag_in_generated_columns(
        self,
    ):
        # Arrange: no zeros in 'b' so no __is_invalid column produced at all —
        # verifies the non-zero-denominator case still works correctly.
        assembler = PipelineAssembler()
        X = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "b": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            }
        )
        params = {"math_ops_1": "('a', 'b', 'divide')"}
        spaces = {
            "math_ops_1": ["none", "('a', 'b', 'divide')"],
            "column_selection_candidates": [["a", "b"]],
        }

        # Act
        dynamic_spaces = assembler._discover_dynamic_spaces(
            params, spaces, model_type="other", X=X, y=None
        )

        # Assert: valid result column discovered, no flag column anywhere
        assert "norm_a__divide__b" in dynamic_spaces
        for key in dynamic_spaces:
            assert "__is_invalid" not in key


class TestDimReductionColumnVisibility:
    def test_should_include_generated_columns_in_dim_reduction_features_after_expansion(
        self,
    ):
        # Arrange: two numeric columns, math-op that generates a third
        assembler = PipelineAssembler()

        X = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "b": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            }
        )

        # Spaces: math-op that produces "a__multiply__b", plus dim-reduction,
        # plus the original column_selection_candidates covering only a and b.
        spaces = {
            "math_ops_1": ["none", "('a', 'b', 'multiply')"],
            "dim_reduction_method": ["none", "pca"],
            "dim_reduction_n_components": [2],
            "column_selection_candidates": [["a", "b"]],
        }

        # _FakeTrial always picks choices[0].  Override so math_ops_1 picks the
        # operation value and dim_reduction_method picks "pca".
        class _SelectingTrial:
            def __init__(self, overrides: dict) -> None:
                self._overrides = overrides
                self.suggested: dict = {}

            def suggest_categorical(self, name: str, choices: list) -> object:
                self.suggested[name] = list(choices)
                return self._overrides.get(name, choices[0])

        trial = _SelectingTrial(
            {
                "math_ops_1": "('a', 'b', 'multiply')",
                "dim_reduction_method": "pca",
                "dim_reduction_n_components": 2,
            }
        )

        # Act
        result = assembler.assemble(trial, spaces, model_type="other", X=X, y=None)

        # Assert: DimensionalityReduction transformer is present
        dim_reduction_dicts = [
            d for d in result if d.get("name") == "DimensionalityReduction"
        ]
        assert len(dim_reduction_dicts) == 1, (
            "Expected exactly one DimensionalityReduction transformer dict"
        )

        features = dim_reduction_dicts[0]["params"]["features"]
        assert features is not None, "features should not be None"
        generated_col = "a__multiply__b"
        assert generated_col in features, (
            f"Generated column '{generated_col}' not found in DimensionalityReduction "
            f"features: {features}"
        )
