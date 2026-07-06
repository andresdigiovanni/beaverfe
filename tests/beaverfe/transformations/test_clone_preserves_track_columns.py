"""Regression test for the systemic get_params()/clone() mismatch (§6.5):
every transformer used to hand-roll get_params()/set_params() and omit
track_columns, so sklearn.base.clone() (used internally by cross_val_score
on every CV fold) silently reset it to False. All hand-rolled overrides were
removed in favor of BaseEstimator's inherited, __init__-signature-based
implementation, which includes every constructor argument automatically.
"""

import pytest
from sklearn.base import clone

from beaverfe.beaver_pipeline import BeaverPipeline
from beaverfe.transformations import (
    CategoricalEncoding,
    ColumnSelection,
    CyclicalFeaturesTransformer,
    DateTimeTransformer,
    DimensionalityReduction,
    MathematicalOperations,
    MissingValuesHandler,
    MissingValuesIndicator,
    NonLinearTransformation,
    Normalization,
    NumericalBinning,
    OutliersHandler,
    QuantileTransformation,
    ScaleTransformation,
    SplineTransformation,
)

TRANSFORMER_CLASSES = [
    CategoricalEncoding,
    ColumnSelection,
    CyclicalFeaturesTransformer,
    DateTimeTransformer,
    DimensionalityReduction,
    MathematicalOperations,
    MissingValuesHandler,
    MissingValuesIndicator,
    NonLinearTransformation,
    Normalization,
    NumericalBinning,
    OutliersHandler,
    QuantileTransformation,
    ScaleTransformation,
    SplineTransformation,
]


@pytest.mark.parametrize("transformer_class", TRANSFORMER_CLASSES)
def test_should_preserve_track_columns_through_clone(transformer_class):
    instance = transformer_class(track_columns=True)

    cloned = clone(instance)

    assert cloned.get_params()["track_columns"] is True


def test_should_preserve_beaver_pipeline_params_through_clone():
    pipeline = BeaverPipeline(
        [{"name": "ColumnSelection", "params": {"features": ["a"]}}]
    )

    cloned = clone(pipeline)

    assert cloned.get_params()["order"] == pipeline.order
