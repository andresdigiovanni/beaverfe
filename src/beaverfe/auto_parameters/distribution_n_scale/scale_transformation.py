import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class ScaleTransformationSpaceGenerator(PerColumnSpaceGenerator):
    prefix = "scale"
    QUANTILE_RANGE_OPTIONS = [25.0]

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        robust_options = [f"robust__{q_low}" for q_low in self.QUANTILE_RANGE_OPTIONS]
        return ["standard", *robust_options]
