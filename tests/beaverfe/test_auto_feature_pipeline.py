from beaverfe.auto_feature_pipeline import (
    _prune_by_selected_transformed,
    _prune_eliminated_columns,
    _recipe_guarantees_clean_output,
    _update_extended_keep,
)


class TestPruneEliminatedColumns:
    def test_should_drop_column_entries_not_in_keep_set(self):
        # Arrange
        transformer_dicts = [
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {"a": "standard", "b": "standard"}
                },
            }
        ]

        # Act
        result = _prune_eliminated_columns(transformer_dicts, keep={"a"})

        # Assert
        assert result[0]["params"]["transformation_options"] == {"a": "standard"}

    def test_should_drop_entire_entry_when_no_columns_survive(self):
        # Arrange
        transformer_dicts = [
            {
                "name": "ScaleTransformation",
                "params": {"transformation_options": {"a": "standard"}},
            }
        ]

        # Act
        result = _prune_eliminated_columns(transformer_dicts, keep={"z"})

        # Assert
        assert result == []

    def test_should_drop_math_operations_referencing_eliminated_columns(self):
        # Arrange
        transformer_dicts = [
            {
                "name": "MathematicalOperations",
                "params": {
                    "operations_options": [
                        ("a", "b", "multiply"),
                        ("a", "c", "divide"),
                    ]
                },
            }
        ]

        # Act
        result = _prune_eliminated_columns(transformer_dicts, keep={"a", "b"})

        # Assert
        assert result[0]["params"]["operations_options"] == [("a", "b", "multiply")]

    def test_should_prune_features_list_when_column_eliminated(self):
        # Arrange
        transformer_dicts = [
            {
                "name": "DimensionalityReduction",
                "params": {"method": "pca", "n_components": 2, "features": ["a", "b"]},
            }
        ]

        # Act
        result = _prune_eliminated_columns(transformer_dicts, keep={"a"})

        # Assert
        assert result[0]["params"]["features"] == ["a"]

    def test_should_leave_entries_unchanged_when_all_columns_kept(self):
        # Arrange
        transformer_dicts = [
            {
                "name": "CategoricalEncoding",
                "params": {"transformation_options": {"gender": "dummy"}},
            }
        ]

        # Act
        result = _prune_eliminated_columns(transformer_dicts, keep={"gender"})

        # Assert
        assert result == transformer_dicts

    def test_should_return_empty_list_when_input_is_empty(self):
        # Arrange & Act
        result = _prune_eliminated_columns([], keep=set())

        # Assert
        assert result == []

    def test_should_keep_generated_math_op_columns_in_downstream_scale_step(self):
        # Arrange: MathematicalOperations generates a__add__b, then a downstream
        # scale step references both original and generated columns.
        transformer_dicts = [
            {
                "name": "MathematicalOperations",
                "params": {
                    "operations_options": [("a", "b", "add"), ("a", "c", "multiply")]
                },
            },
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {
                        "a": "standard",
                        "b": "standard",
                        "c": "standard",
                        "a__add__b": "standard",
                        "a__multiply__c": "standard",
                    }
                },
            },
        ]

        # Act: keep only a and b (drop c)
        result = _prune_eliminated_columns(transformer_dicts, keep={"a", "b"})

        # Assert: math op a+b survives; a*c is dropped since c is eliminated
        assert result[0]["params"]["operations_options"] == [("a", "b", "add")]
        scale_opts = result[1]["params"]["transformation_options"]
        assert "a__add__b" in scale_opts
        assert "a__multiply__c" not in scale_opts
        assert "c" not in scale_opts

    def test_should_keep_binning_generated_columns_in_downstream_scale_step(self):
        # Arrange
        transformer_dicts = [
            {
                "name": "NumericalBinning",
                "params": {
                    "transformation_options": {
                        "a": ("quantile", 5),
                        "b": ("quantile", 5),
                    }
                },
            },
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {
                        "a__bins_quantile_5": "standard",
                        "b__bins_quantile_5": "standard",
                    }
                },
            },
        ]

        # Act: drop b
        result = _prune_eliminated_columns(transformer_dicts, keep={"a"})

        # Assert
        scale_opts = result[1]["params"]["transformation_options"]
        assert "a__bins_quantile_5" in scale_opts
        assert "b__bins_quantile_5" not in scale_opts

    def test_should_keep_cyclical_generated_columns_in_downstream_scale_step(self):
        # Arrange
        transformer_dicts = [
            {
                "name": "CyclicalFeaturesTransformer",
                "params": {"transformation_options": {"hour": 24, "month": 12}},
            },
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {
                        "hour_sin": "standard",
                        "hour_cos": "standard",
                        "month_sin": "standard",
                        "month_cos": "standard",
                    }
                },
            },
        ]

        # Act: drop month
        result = _prune_eliminated_columns(transformer_dicts, keep={"hour"})

        # Assert
        scale_opts = result[1]["params"]["transformation_options"]
        assert "hour_sin" in scale_opts
        assert "hour_cos" in scale_opts
        assert "month_sin" not in scale_opts
        assert "month_cos" not in scale_opts

    def test_should_keep_missing_indicator_generated_columns_in_downstream_scale_step(
        self,
    ):
        # Arrange
        transformer_dicts = [
            {
                "name": "MissingValuesIndicator",
                "params": {"features": ["a", "b"]},
            },
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {
                        "a__is_missing": "standard",
                        "b__is_missing": "standard",
                    }
                },
            },
        ]

        # Act: drop b
        result = _prune_eliminated_columns(transformer_dicts, keep={"a"})

        # Assert
        scale_opts = result[1]["params"]["transformation_options"]
        assert "a__is_missing" in scale_opts
        assert "b__is_missing" not in scale_opts


class TestUpdateExtendedKeep:
    def test_missing_indicator_adds_is_missing_columns(self):
        extended_keep: set[str] = {"a", "b"}
        params = {"features": ["a", "b"]}
        _update_extended_keep("MissingValuesIndicator", params, extended_keep)
        assert "a__is_missing" in extended_keep
        assert "b__is_missing" in extended_keep

    def test_datetime_transformer_adds_all_suffixes(self):
        extended_keep: set[str] = {"ts"}
        params = {"features": ["ts"]}
        _update_extended_keep("DateTimeTransformer", params, extended_keep)
        for suffix in (
            "_year",
            "_month",
            "_day",
            "_weekday",
            "_hour",
            "_minute",
            "_second",
        ):
            assert f"ts{suffix}" in extended_keep

    def test_cyclical_transformer_adds_sin_cos(self):
        extended_keep: set[str] = {"hour"}
        params = {"transformation_options": {"hour": 24}}
        _update_extended_keep("CyclicalFeaturesTransformer", params, extended_keep)
        assert "hour_sin" in extended_keep
        assert "hour_cos" in extended_keep

    def test_math_operations_adds_generated_and_invalid_columns(self):
        extended_keep: set[str] = {"a", "b"}
        params = {"operations_options": [("a", "b", "divide")]}
        _update_extended_keep("MathematicalOperations", params, extended_keep)
        assert "a__divide__b" in extended_keep
        assert "a__divide__b__is_invalid" in extended_keep

    def test_numerical_binning_adds_binned_column_name(self):
        extended_keep: set[str] = {"x"}
        params = {"transformation_options": {"x": ("quantile", 10)}}
        _update_extended_keep("NumericalBinning", params, extended_keep)
        assert "x__bins_quantile_10" in extended_keep


class TestPruneBySelectedTransformed:
    ALL_COLS = {"a", "b", "c", "a__add__b", "a__bins_quantile_5", "a_sin", "a_cos"}

    def test_math_ops_removes_operations_not_in_selected(self):
        dicts = [
            {
                "name": "MathematicalOperations",
                "params": {
                    "operations_options": [
                        ("a", "b", "add"),
                        ("a", "c", "multiply"),
                    ]
                },
            }
        ]
        selected = {"a__add__b"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert result[0]["params"]["operations_options"] == [("a", "b", "add")]

    def test_math_ops_step_removed_when_all_ops_eliminated(self):
        dicts = [
            {
                "name": "MathematicalOperations",
                "params": {"operations_options": [("a", "b", "add")]},
            }
        ]
        selected = {"a", "b"}  # math output not selected
        result = _prune_by_selected_transformed(dicts, selected)
        assert result == []

    def test_numerical_binning_removes_cols_not_in_selected(self):
        dicts = [
            {
                "name": "NumericalBinning",
                "params": {
                    "transformation_options": {
                        "a": ("quantile", 5),
                        "b": ("quantile", 5),
                    }
                },
            }
        ]
        selected = {"a__bins_quantile_5"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert "a" in result[0]["params"]["transformation_options"]
        assert "b" not in result[0]["params"]["transformation_options"]

    def test_spline_removes_cols_with_no_surviving_splines(self):
        dicts = [
            {
                "name": "SplineTransformation",
                "params": {
                    "transformation_options": {
                        "a": {"n_knots": 3, "degree": 3},
                        "b": {"n_knots": 3, "degree": 3},
                    }
                },
            }
        ]
        selected = {"a__spline_0", "a__spline_1"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert "a" in result[0]["params"]["transformation_options"]
        assert "b" not in result[0]["params"]["transformation_options"]

    def test_inplace_step_removes_entries_for_non_needed_columns(self):
        # 'c' is not selected and not needed by any expansion step → removed
        dicts = [
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {"a": "standard", "c": "standard"},
                    "quantile_range": {},
                },
            }
        ]
        selected = {"a"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert result[0]["params"]["transformation_options"] == {"a": "standard"}
        assert "c" not in result[0]["params"]["transformation_options"]

    def test_inplace_step_keeps_col_needed_by_surviving_expansion(self):
        # 'a' itself is NOT in selected_transformed but feeds a surviving NumericalBinning
        dicts = [
            {
                "name": "NumericalBinning",
                "params": {"transformation_options": {"a": ("quantile", 5)}},
            },
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {"a": "standard"},
                    "quantile_range": {},
                },
            },
        ]
        # a__bins_quantile_5 survived; 'a' itself did not appear in selected
        selected = {"a__bins_quantile_5"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert result[1]["params"]["transformation_options"] == {"a": "standard"}

    def test_dimensionality_reduction_kept_when_output_col_selected(self):
        dicts = [
            {
                "name": "DimensionalityReduction",
                "params": {
                    "method": "lda",
                    "n_components": 2,
                    "features": ["a", "b", "c"],
                },
            }
        ]
        selected = {"lda_1", "lda_2"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert result[0]["name"] == "DimensionalityReduction"
        assert result[0]["params"]["features"] == ["a", "b", "c"]

    def test_dimensionality_reduction_removed_when_no_output_selected(self):
        dicts = [
            {
                "name": "DimensionalityReduction",
                "params": {"method": "pca", "n_components": 2, "features": ["a", "b"]},
            }
        ]
        selected = {"a", "b"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert result == []

    def test_outliers_handler_extra_dicts_pruned(self):
        dicts = [
            {
                "name": "OutliersHandler",
                "params": {
                    "transformation_options": {
                        "a": ("cap", "iqr"),
                        "b": ("median", "iforest"),
                    },
                    "thresholds": {"a": 1.5},
                    "iforest_params": {"b": {"contamination": 0.05}},
                },
            }
        ]
        selected = {"a"}
        result = _prune_by_selected_transformed(dicts, selected)
        assert "b" not in result[0]["params"]["transformation_options"]
        assert "b" not in result[0]["params"]["iforest_params"]

    def test_lda_pipeline_prunes_binning_and_math_ops(self):
        # Simulates the wine dataset case: LDA uses all originals.
        # SplineTransformation columns were not selected → remove spline entry.
        # NumericalBinning for 'b' was not selected → remove it.
        # OutliersHandler for 'b__bins_q_5' (generated) → removed.
        # OutliersHandler for 'a', 'b' (originals feeding LDA) → kept.
        dicts = [
            {
                "name": "NumericalBinning",
                "params": {
                    "transformation_options": {
                        "a": ("quantile", 5),
                        "b": ("quantile", 5),
                    }
                },
            },
            {
                "name": "OutliersHandler",
                "params": {
                    "transformation_options": {
                        "a": ("cap", "iqr"),
                        "b": ("cap", "iqr"),
                        "a__bins_quantile_5": ("median", "iforest"),
                        "b__bins_quantile_5": ("median", "iforest"),
                    },
                    "thresholds": {"a": 1.5, "b": 1.5},
                    "iforest_params": {
                        "a__bins_quantile_5": {"contamination": 0.05},
                        "b__bins_quantile_5": {"contamination": 0.05},
                    },
                },
            },
            {
                "name": "DimensionalityReduction",
                "params": {
                    "method": "lda",
                    "n_components": 1,
                    "features": ["a", "b"],
                },
            },
        ]
        # Only a__bins_quantile_5 and lda_1 survived RFECV
        selected = {"a__bins_quantile_5", "lda_1"}
        result = _prune_by_selected_transformed(dicts, selected)

        names = [d["name"] for d in result]
        assert "NumericalBinning" in names
        assert "DimensionalityReduction" in names

        # NumericalBinning: only 'a' survived
        nb = next(d for d in result if d["name"] == "NumericalBinning")
        assert "a" in nb["params"]["transformation_options"]
        assert "b" not in nb["params"]["transformation_options"]

        # OutliersHandler: 'a' and 'b' kept (feed LDA), 'a__bins_quantile_5' kept
        # (survived), 'b__bins_quantile_5' removed (not selected, not needed)
        oh = next(d for d in result if d["name"] == "OutliersHandler")
        assert "a" in oh["params"]["transformation_options"]
        assert "b" in oh["params"]["transformation_options"]
        assert "a__bins_quantile_5" in oh["params"]["transformation_options"]
        assert "b__bins_quantile_5" not in oh["params"]["transformation_options"]
        assert "b__bins_quantile_5" not in oh["params"]["iforest_params"]


class TestRecipeGuaranteesCleanOutput:
    """Unit tests for the _recipe_guarantees_clean_output helper."""

    def test_should_return_true_when_missing_handler_present_and_no_cat_enc_keys_in_spaces(
        self,
    ):
        # (a) MissingValuesHandler present, no cat_enc_* keys in spaces
        # -> no categorical columns existed, so absence of CategoricalEncoding is fine
        transformer_dicts = [{"name": "MissingValuesHandler", "params": {}}]
        spaces = {"scale_a": ["none", "standard"], "impute_a": ["mean"]}

        result = _recipe_guarantees_clean_output(transformer_dicts, spaces)

        assert result is True

    def test_should_return_false_when_missing_handler_present_but_cat_enc_key_exists_without_encoding_step(
        self,
    ):
        # (b) MissingValuesHandler present, cat_enc_ keys exist, but no CategoricalEncoding in recipe
        transformer_dicts = [{"name": "MissingValuesHandler", "params": {}}]
        spaces = {
            "cat_enc_gender": ["dummy", "catboost"],
            "impute_a": ["mean"],
        }

        result = _recipe_guarantees_clean_output(transformer_dicts, spaces)

        assert result is False

    def test_should_return_true_when_both_missing_handler_and_categorical_encoding_present(
        self,
    ):
        # (c) Both MissingValuesHandler and CategoricalEncoding present in recipe
        transformer_dicts = [
            {"name": "MissingValuesHandler", "params": {}},
            {"name": "CategoricalEncoding", "params": {}},
        ]
        spaces = {
            "cat_enc_gender": ["dummy", "catboost"],
            "impute_a": ["mean"],
        }

        result = _recipe_guarantees_clean_output(transformer_dicts, spaces)

        assert result is True

    def test_should_return_false_when_missing_handler_absent(self):
        # MissingValuesHandler absent -> always False regardless of categorical encoding
        transformer_dicts = [{"name": "CategoricalEncoding", "params": {}}]
        spaces = {}

        result = _recipe_guarantees_clean_output(transformer_dicts, spaces)

        assert result is False

    def test_should_return_false_when_recipe_is_empty(self):
        result = _recipe_guarantees_clean_output([], spaces={})

        assert result is False


class TestPruneBySelectedTransformedFullPipeline:
    """Tests for _prune_by_selected_transformed with realistic full-pipeline scenarios."""

    WINE_ORIGINALS = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavonoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315_of_diluted_wines",
        "proline",
    ]

    def _make_wine_recipe(self) -> list[dict]:
        """Construct a realistic wine-dataset recipe with all major transformer types."""
        originals = self.WINE_ORIGINALS
        return [
            {
                "name": "MathematicalOperations",
                "params": {
                    "operations_options": [
                        ("flavanoids", "alcohol", "multiply"),
                        ("malic_acid", "ash", "add"),
                        ("total_phenols", "proline", "divide"),
                    ]
                },
            },
            {
                "name": "SplineTransformation",
                "params": {
                    "transformation_options": {
                        "malic_acid": {"n_knots": 4, "degree": 3},
                        "alcohol": {"n_knots": 4, "degree": 3},
                        "ash": {"n_knots": 4, "degree": 3},
                    }
                },
            },
            {
                "name": "NumericalBinning",
                "params": {
                    "transformation_options": dict.fromkeys(
                        ["alcohol", "malic_acid", "ash"], ("quantile", 10)
                    )
                },
            },
            {
                "name": "OutliersHandler",
                "params": {
                    "transformation_options": dict.fromkeys(originals, ("cap", "iqr")),
                    "thresholds": dict.fromkeys(originals, 1.5),
                    "iforest_params": {},
                    "lof_params": {},
                },
            },
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": dict.fromkeys(originals, "standard"),
                },
            },
            {
                "name": "DimensionalityReduction",
                "params": {
                    "method": "lda",
                    "n_components": 2,
                    "features": list(originals),
                },
            },
        ]

    def test_should_remove_math_ops_when_no_math_output_selected(self):
        # Arrange
        dicts = self._make_wine_recipe()

        # Act: only spline and LDA outputs survive
        selected = {"malic_acid__spline_10", "malic_acid__spline_11", "lda_1", "lda_2"}
        result = _prune_by_selected_transformed(dicts, selected)

        # Assert: MathematicalOperations removed entirely
        names = [d["name"] for d in result]
        assert "MathematicalOperations" not in names

    def test_should_remove_numerical_binning_when_no_bin_output_selected(self):
        # Arrange
        dicts = self._make_wine_recipe()

        # Act
        selected = {"malic_acid__spline_10", "malic_acid__spline_11", "lda_1", "lda_2"}
        result = _prune_by_selected_transformed(dicts, selected)

        # Assert: NumericalBinning removed entirely
        names = [d["name"] for d in result]
        assert "NumericalBinning" not in names

    def test_should_keep_only_malic_acid_in_spline_when_only_its_output_selected(self):
        # Arrange
        dicts = self._make_wine_recipe()

        # Act
        selected = {"malic_acid__spline_10", "malic_acid__spline_11", "lda_1", "lda_2"}
        result = _prune_by_selected_transformed(dicts, selected)

        # Assert: SplineTransformation kept, only malic_acid entry remains
        names = [d["name"] for d in result]
        assert "SplineTransformation" in names
        spline = next(d for d in result if d["name"] == "SplineTransformation")
        assert list(spline["params"]["transformation_options"].keys()) == ["malic_acid"]

    def test_should_keep_dim_reduction_with_all_original_features_when_lda_output_selected(
        self,
    ):
        # Arrange
        dicts = self._make_wine_recipe()

        # Act
        selected = {"malic_acid__spline_10", "malic_acid__spline_11", "lda_1", "lda_2"}
        result = _prune_by_selected_transformed(dicts, selected)

        # Assert: DimensionalityReduction kept with all 13 original features
        names = [d["name"] for d in result]
        assert "DimensionalityReduction" in names
        dr = next(d for d in result if d["name"] == "DimensionalityReduction")
        assert len(dr["params"]["features"]) == len(self.WINE_ORIGINALS)
        assert set(dr["params"]["features"]) == set(self.WINE_ORIGINALS)

    def test_should_retain_all_original_cols_in_outliers_handler_when_lda_needs_them(
        self,
    ):
        # Arrange
        dicts = self._make_wine_recipe()

        # Act: LDA outputs selected → all 13 originals are needed
        selected = {"malic_acid__spline_10", "malic_acid__spline_11", "lda_1", "lda_2"}
        result = _prune_by_selected_transformed(dicts, selected)

        # Assert: OutliersHandler keeps exactly the 13 original columns
        oh = next(d for d in result if d["name"] == "OutliersHandler")
        oh_cols = set(oh["params"]["transformation_options"].keys())
        assert oh_cols == set(self.WINE_ORIGINALS)

    def test_should_retain_all_original_cols_in_scale_when_lda_needs_them(self):
        # Arrange
        dicts = self._make_wine_recipe()

        # Act
        selected = {"lda_1", "lda_2"}
        result = _prune_by_selected_transformed(dicts, selected)

        # Assert: ScaleTransformation keeps all 13 originals (needed by LDA)
        scale = next(d for d in result if d["name"] == "ScaleTransformation")
        scale_cols = set(scale["params"]["transformation_options"].keys())
        assert scale_cols == set(self.WINE_ORIGINALS)

    def test_should_return_empty_list_when_recipe_has_only_in_place_steps_and_nothing_needed(
        self,
    ):
        # Arrange: recipe with only an in-place step referencing columns not needed
        dicts = [
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {"a": "standard", "b": "standard"},
                },
            }
        ]

        # Act: selected set is disjoint from recipe columns
        selected = {"x_0", "x_1"}  # none of a or b are needed
        result = _prune_by_selected_transformed(dicts, selected)

        # Assert: empty list returned — no crash
        assert result == []

    def test_should_handle_empty_recipe_without_crash(self):
        # Arrange & Act
        result = _prune_by_selected_transformed([], selected_transformed={"lda_1"})

        # Assert
        assert result == []

    def test_should_handle_empty_selected_set_with_in_place_only_recipe(self):
        # Arrange
        dicts = [
            {
                "name": "ScaleTransformation",
                "params": {"transformation_options": {"a": "standard"}},
            }
        ]

        # Act: empty selection means nothing is needed
        result = _prune_by_selected_transformed(dicts, selected_transformed=set())

        # Assert: all in-place entries removed since needed_columns is empty
        assert result == []

    def test_should_keep_categorical_encoding_when_expansion_encoded_columns_selected(
        self,
    ):
        # Arrange: regression for the Adult-dataset KeyError bug.
        # CategoricalEncoding produces expansion columns like `marital_status_1`;
        # the downstream ColumnSelection references those encoded names.
        # _prune_by_selected_transformed must NOT drop CategoricalEncoding
        # when only encoded (not original) column names appear in selected_transformed.
        dicts = [
            {
                "name": "CategoricalEncoding",
                "params": {
                    "transformation_options": {
                        "marital_status": "onehot",
                        "occupation": "onehot",
                        "sex": "label",  # in-place — selected by original name
                    }
                },
            },
            {
                "name": "ScaleTransformation",
                "params": {
                    "transformation_options": {"age": "standard"},
                },
            },
        ]

        # Encoded expansion columns + in-place label column + a raw numeric column
        selected = {
            "marital_status_1",
            "marital_status_3",
            "occupation_2",
            "sex",
            "age",
        }

        result = _prune_by_selected_transformed(dicts, selected)

        # CategoricalEncoding must be retained with all three original columns
        names = [d["name"] for d in result]
        assert "CategoricalEncoding" in names

        cat_enc = next(d for d in result if d["name"] == "CategoricalEncoding")
        enc_opts = cat_enc["params"]["transformation_options"]
        assert "marital_status" in enc_opts  # expansion encoder needed
        assert "occupation" in enc_opts  # expansion encoder needed
        assert "sex" in enc_opts  # in-place encoder needed

    def test_should_drop_categorical_encoding_column_when_no_encoded_output_selected(
        self,
    ):
        # Arrange: only the `sex` column (in-place label) is NOT selected;
        # occupation's encoded columns are not in selected either.
        dicts = [
            {
                "name": "CategoricalEncoding",
                "params": {
                    "transformation_options": {
                        "occupation": "onehot",
                        "sex": "label",
                    }
                },
            },
        ]

        selected = {"marital_status_1", "age"}  # neither occupation_* nor sex

        result = _prune_by_selected_transformed(dicts, selected)

        # CategoricalEncoding should be dropped entirely
        names = [d["name"] for d in result]
        assert "CategoricalEncoding" not in names

    def test_should_keep_math_op_when_only_is_invalid_companion_selected(self):
        # Regression for the Credit-dataset KeyError bug.
        # AGE__divide__PAY_0__is_invalid may be selected by permutation
        # importance while AGE__divide__PAY_0 itself is not. The MathOp
        # must be retained so the __is_invalid column is actually produced.
        dicts = [
            {
                "name": "MathematicalOperations",
                "params": {
                    "operations_options": [
                        ("AGE", "PAY_0", "divide"),
                        ("A", "B", "multiply"),
                    ]
                },
            }
        ]

        # Only the __is_invalid companion is selected; primary output is not.
        selected = {"AGE__divide__PAY_0__is_invalid", "some_other_col"}

        result = _prune_by_selected_transformed(dicts, selected)

        names = [d["name"] for d in result]
        assert "MathematicalOperations" in names

        ops = result[0]["params"]["operations_options"]
        assert ("AGE", "PAY_0", "divide") in ops
        assert ("A", "B", "multiply") not in ops  # no output selected for this one

    def test_should_keep_math_op_when_primary_output_selected(self):
        # Primary output selected (not just __is_invalid) — baseline behaviour.
        dicts = [
            {
                "name": "MathematicalOperations",
                "params": {"operations_options": [("A", "B", "add")]},
            }
        ]

        selected = {"A__add__B"}

        result = _prune_by_selected_transformed(dicts, selected)

        ops = result[0]["params"]["operations_options"]
        assert ("A", "B", "add") in ops

    def test_should_clamp_n_components_when_features_pruned_below_original_count(self):
        # Regression for "n_components(10) must be <= n_features(6)".
        # After pruning, DimensionalityReduction may have fewer features than
        # the n_components chosen during optimisation; clamp it.
        dicts = [
            {
                "name": "DimensionalityReduction",
                "params": {
                    "method": "pca",
                    "n_components": 10,
                    "features": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                },
            }
        ]

        # Only 6 of the 10 pca outputs survive selection
        selected = {"pca_1", "pca_2", "pca_3", "pca_4", "pca_5", "pca_6"}

        result = _prune_by_selected_transformed(dicts, selected)

        assert len(result) == 1
        dr = result[0]
        assert dr["params"]["n_components"] <= len(dr["params"]["features"])
