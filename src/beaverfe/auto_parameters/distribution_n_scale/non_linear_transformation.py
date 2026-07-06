import pandas as pd
from scipy.stats import skew

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class NonLinearTransformationSpaceGenerator(PerColumnSpaceGenerator):
    prefix = "nonlinear"
    SKEWNESS_THRESHOLD = 0.5

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str] | None:
        col_skew = float(skew(X[col].dropna()))
        if abs(col_skew) >= self.SKEWNESS_THRESHOLD:
            return ["yeo_johnson"]
        return None
