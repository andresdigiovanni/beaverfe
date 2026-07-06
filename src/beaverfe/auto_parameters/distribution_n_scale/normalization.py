import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class NormalizationSpaceGenerator(PerColumnSpaceGenerator):
    prefix = "normalize"
    NORMALIZATION_OPTIONS = ["l2"]

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        return self.NORMALIZATION_OPTIONS
