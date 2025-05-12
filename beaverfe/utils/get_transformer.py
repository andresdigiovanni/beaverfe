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


def get_transformer(name, params):
    transformer_mapping = {
        "CategoricalEncoding": CategoricalEncoding,
        "ColumnSelection": ColumnSelection,
        "CyclicalFeaturesTransformer": CyclicalFeaturesTransformer,
        "DateTimeTransformer": DateTimeTransformer,
        "DimensionalityReduction": DimensionalityReduction,
        "MathematicalOperations": MathematicalOperations,
        "MissingValuesHandler": MissingValuesHandler,
        "MissingValuesIndicator": MissingValuesIndicator,
        "NonLinearTransformation": NonLinearTransformation,
        "Normalization": Normalization,
        "NumericalBinning": NumericalBinning,
        "OutliersHandler": OutliersHandler,
        "QuantileTransformation": QuantileTransformation,
        "ScaleTransformation": ScaleTransformation,
        "SplineTransformation": SplineTransformation,
    }

    if name in transformer_mapping:
        return transformer_mapping[name](**params)

    raise ValueError(f"Unknown transformer: {name}")
