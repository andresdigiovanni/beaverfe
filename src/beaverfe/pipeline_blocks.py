from enum import StrEnum


class PipelineBlock(StrEnum):
    MISSING_INDICATOR = "missing_indicator"
    MISSING_HANDLER = "missing_handler"
    EXTRACTION = "extraction"
    EXPAND_CYCLICAL = "expand_cyclical"
    CLEANSE_OUTLIERS = "cleanse_outliers"
    EXPAND_MATH = "expand_math"
    EXPAND_SPLINE = "expand_spline"
    EXPAND_BINNING = "expand_binning"
    ENCODE = "encode"
    NORMALIZE = "normalize"
    REDUCE_DIM = "reduce_dim"


# v3 canonical order rationale:
#   1. EXTRACTION first: date/JSON parsing may produce new NaNs that
#      MISSING_INDICATOR/MISSING_HANDLER must catch immediately after.
#   2. CLEANSE_OUTLIERS before EXPAND_MATH/EXPAND_SPLINE: avoids outlier
#      amplification (e.g. squaring an outlier explodes it).
#   3. EXPAND_BINNING just before ENCODE: binning produces categoricals
#      that must pass through encoding alongside the rest.
#   4. NORMALIZE is a single exclusive choice per column replacing the old
#      sequential DISTRIBUTE → SCALE → QUANTILE trio, which was incoherent
#      because QUANTILE remapped all data and destroyed prior work.
# Column selection has no dedicated block: PipelineAssembler never produces
# one (it's permanently a no-op), since the real selection mechanism is the
# post-hoc permutation-importance pruning step in auto_feature_pipeline.py.
CANONICAL_ORDER: list[PipelineBlock] = [
    PipelineBlock.EXTRACTION,
    PipelineBlock.MISSING_INDICATOR,
    PipelineBlock.MISSING_HANDLER,
    PipelineBlock.EXPAND_CYCLICAL,
    PipelineBlock.CLEANSE_OUTLIERS,
    PipelineBlock.EXPAND_MATH,
    PipelineBlock.EXPAND_SPLINE,
    PipelineBlock.EXPAND_BINNING,
    PipelineBlock.ENCODE,
    PipelineBlock.NORMALIZE,
    PipelineBlock.REDUCE_DIM,
]

# `auto_feature_pipeline` strips "block" tags before returning a recipe (its
# public format is just {"name", "params"}), so a transformer dict with no
# explicit "block" key still needs a way to be placed correctly by
# BeaverPipeline(recipe, order=CANONICAL_ORDER). Every transformer name below
# is owned by exactly one block, so this mapping lets ordering fall back to
# the name instead of silently treating every block-less entry as "last".
# "ColumnSelection" is deliberately absent: it has no dedicated block (see
# CANONICAL_ORDER's comment) and manually-added column-selection steps
# should run last anyway, which is exactly the fallback for an unmapped name.
TRANSFORMER_NAME_TO_BLOCK: dict[str, PipelineBlock] = {
    "MissingValuesIndicator": PipelineBlock.MISSING_INDICATOR,
    "MissingValuesHandler": PipelineBlock.MISSING_HANDLER,
    "DateTimeTransformer": PipelineBlock.EXTRACTION,
    "CyclicalFeaturesTransformer": PipelineBlock.EXPAND_CYCLICAL,
    "MathematicalOperations": PipelineBlock.EXPAND_MATH,
    "SplineTransformation": PipelineBlock.EXPAND_SPLINE,
    "NumericalBinning": PipelineBlock.EXPAND_BINNING,
    "OutliersHandler": PipelineBlock.CLEANSE_OUTLIERS,
    "CategoricalEncoding": PipelineBlock.ENCODE,
    "NonLinearTransformation": PipelineBlock.NORMALIZE,
    "Normalization": PipelineBlock.NORMALIZE,
    "ScaleTransformation": PipelineBlock.NORMALIZE,
    "QuantileTransformation": PipelineBlock.NORMALIZE,
    "DimensionalityReduction": PipelineBlock.REDUCE_DIM,
}
