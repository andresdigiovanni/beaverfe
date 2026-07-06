import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class QuantileTransformationSpaceGenerator(PerColumnSpaceGenerator):
    prefix = "quantile"
    TRANSFORMATION_OPTIONS = ["uniform", "normal"]

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        return self.TRANSFORMATION_OPTIONS
