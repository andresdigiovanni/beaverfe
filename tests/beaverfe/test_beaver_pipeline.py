"""Unit tests for BeaverPipeline dynamic block ordering."""

import pandas as pd

from beaverfe.beaver_pipeline import BeaverPipeline
from beaverfe.pipeline_blocks import CANONICAL_ORDER, PipelineBlock

# ---------------------------------------------------------------------------
# Minimal transformer dicts for testing ordering (no real sklearn transformers)
# ---------------------------------------------------------------------------


def _make_transformer_dict(name: str, block: PipelineBlock | None = None) -> dict:
    d: dict = {"name": name, "params": {}}
    if block is not None:
        d["block"] = block
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBeaverPipelineOrdering:
    """Test that BeaverPipeline respects the optional block ordering."""

    def test_should_preserve_list_order_when_no_block_keys_and_no_order(self):
        """Legacy behaviour: no 'block' keys, no order -> transformers run in list order."""
        # Use a list of dicts without 'block' keys — order must be preserved.
        transformations = [
            {"name": "ColumnSelection", "params": {"features": ["a"]}},
            {"name": "ColumnSelection", "params": {"features": ["a"]}},
        ]
        pipeline = BeaverPipeline(transformations)
        # The initialised transformers list should be the same length (not reordered).
        assert len(pipeline.transformations) == 2

    def test_should_sort_by_block_when_order_is_provided(self):
        """Transformers with 'block' keys are sorted according to the provided order."""
        # Provide dicts in reverse canonical order; the pipeline should sort them.
        order = [
            PipelineBlock.MISSING_INDICATOR,
            PipelineBlock.MISSING_HANDLER,
            PipelineBlock.ENCODE,
        ]
        transformations = [
            _make_transformer_dict("ColumnSelection", PipelineBlock.ENCODE),
            _make_transformer_dict(
                "MissingValuesIndicator", PipelineBlock.MISSING_INDICATOR
            ),
            _make_transformer_dict(
                "MissingValuesHandler", PipelineBlock.MISSING_HANDLER
            ),
        ]
        # _sort_by_order is called in __init__ when order is provided
        sorted_dicts = BeaverPipeline([], order=order)._sort_by_order(
            transformations, order
        )
        blocks = [d["block"] for d in sorted_dicts]
        assert blocks == [
            PipelineBlock.MISSING_INDICATOR,
            PipelineBlock.MISSING_HANDLER,
            PipelineBlock.ENCODE,
        ]

    def test_should_append_blockless_dicts_last_when_mixed_with_block_dicts(self):
        """Dicts without a 'block' key are appended after all keyed dicts."""
        order = [PipelineBlock.MISSING_INDICATOR, PipelineBlock.ENCODE]
        transformations = [
            {"name": "NoBlock", "params": {}},  # no 'block' key
            _make_transformer_dict("CategoricalEncoding", PipelineBlock.ENCODE),
            _make_transformer_dict(
                "MissingValuesIndicator", PipelineBlock.MISSING_INDICATOR
            ),
        ]
        sorted_dicts = BeaverPipeline([], order=order)._sort_by_order(
            transformations, order
        )
        # Last entry should be the one without a block key
        assert sorted_dicts[-1]["name"] == "NoBlock"
        assert sorted_dicts[0]["name"] == "MissingValuesIndicator"

    def test_should_preserve_original_order_when_empty_order_provided(self):
        """Empty order=[] -> original list order is preserved."""
        transformations = [
            _make_transformer_dict("B", PipelineBlock.ENCODE),
            _make_transformer_dict("A", PipelineBlock.MISSING_INDICATOR),
        ]
        sorted_dicts = BeaverPipeline([], order=[])._sort_by_order(transformations, [])
        # With empty order, sorted returns same input (len == 0 order -> all go to end bucket)
        # Names should be preserved in original order
        assert [d["name"] for d in sorted_dicts] == ["B", "A"]

    def test_should_infer_block_from_name_when_block_key_is_missing(self):
        """A block-less dict (e.g. from auto_feature_pipeline's stripped
        recipe) still sorts correctly via its transformer name."""
        order = [PipelineBlock.MISSING_INDICATOR, PipelineBlock.ENCODE]
        transformations = [
            {"name": "CategoricalEncoding", "params": {}},  # no 'block' key
            {"name": "MissingValuesIndicator", "params": {}},  # no 'block' key
        ]
        sorted_dicts = BeaverPipeline([], order=order)._sort_by_order(
            transformations, order
        )
        assert [d["name"] for d in sorted_dicts] == [
            "MissingValuesIndicator",
            "CategoricalEncoding",
        ]

    def test_should_contain_exactly_11_blocks_when_using_canonical_order(self):
        """CANONICAL_ORDER must contain exactly 11 blocks."""
        assert len(CANONICAL_ORDER) == 11

    def test_should_run_fit_transform_without_error_when_real_data_with_order(self):
        """Smoke test: BeaverPipeline with order runs fit_transform without error."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.Series([0, 1, 0, 1, 0])
        transformations = [
            {
                "name": "ColumnSelection",
                "params": {"features": ["a"]},
            }
        ]
        pipeline = BeaverPipeline(transformations, order=CANONICAL_ORDER)
        result = pipeline.fit_transform(X, y)
        assert list(result.columns) == ["a"]

    def test_should_not_mutate_caller_dataframe_when_fit_transform_runs(self):
        """Regression test: BeaverPipeline no longer copies X itself,
        relying on each transformer's own defensive copy — the caller's
        original DataFrame must still come out untouched."""
        X = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
        X_original = X.copy()
        y = pd.Series([0, 1, 0, 1])
        transformations = [
            {
                "name": "MissingValuesHandler",
                "params": {"transformation_options": {"a": "mean"}},
                "block": PipelineBlock.MISSING_HANDLER,
            }
        ]

        pipeline = BeaverPipeline(transformations, order=CANONICAL_ORDER)
        pipeline.fit_transform(X, y)

        pd.testing.assert_frame_equal(X, X_original)
