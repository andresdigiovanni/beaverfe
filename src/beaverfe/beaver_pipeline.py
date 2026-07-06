from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

from beaverfe.pipeline_blocks import (
    CANONICAL_ORDER,
    TRANSFORMER_NAME_TO_BLOCK,
    PipelineBlock,
)
from beaverfe.utils import get_transformer


class BeaverPipeline(BaseEstimator, TransformerMixin):
    # `transformations` is typed as Any because __init__ accepts list[dict] | None
    # but replaces it with a list of initialised sklearn transformers at runtime.
    transformations: Any

    def __init__(
        self,
        transformations: list[dict] | None = None,
        order: list[PipelineBlock] | None = None,
    ):
        self.transformations = transformations
        self.order = order

        if isinstance(transformations, list) and all(
            isinstance(t, dict) for t in transformations
        ):
            sorted_transformations = (
                self._sort_by_order(transformations, order)
                if order is not None
                else transformations
            )
            self.transformations = self._initialize_transformations(
                sorted_transformations
            )

    def _sort_by_order(
        self,
        transformations: list[dict],
        order: list[PipelineBlock],
    ) -> list[dict]:
        if not order:
            return transformations

        order_index = {block: i for i, block in enumerate(order)}

        def _key(t: dict) -> int:
            block = t.get("block")
            if block is None:
                # No explicit tag (e.g. a recipe returned by
                # auto_feature_pipeline, which strips "block" keys) — fall
                # back to the transformer's name, which owns exactly one block.
                name = t.get("name")
                block = (
                    TRANSFORMER_NAME_TO_BLOCK.get(name)
                    if isinstance(name, str)
                    else None
                )
            if block is None:
                return len(order)
            if isinstance(block, PipelineBlock):
                return order_index.get(block, len(order))
            # Handle string values of block
            try:
                return order_index.get(PipelineBlock(block), len(order))
            except ValueError:
                return len(order)

        return sorted(transformations, key=_key)

    def _initialize_transformations(self, transformations: list[dict]) -> list:
        initialized_transformers = []
        for transformation in transformations:
            transformer = get_transformer(
                transformation["name"], transformation["params"]
            )
            initialized_transformers.append(transformer)
        return initialized_transformers

    def fit(self, X, y=None):
        # No copy here: every transformer already copies internally before
        # mutating, so the caller's X is never touched as it flows through
        # the chain — an extra copy at this level would be pure overhead.
        for transformer in self.transformations:
            transformer.fit(X, y)
            X = transformer.transform(X)
        return self

    def transform(self, X, y=None):
        for transformer in self.transformations:
            X = transformer.transform(X)
        return X

    def fit_transform(self, X, y=None):
        for transformer in self.transformations:
            X = transformer.fit_transform(X, y)
        return X


__all__ = ["CANONICAL_ORDER", "BeaverPipeline", "PipelineBlock"]
