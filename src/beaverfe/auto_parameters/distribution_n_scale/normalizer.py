import pandas as pd
from scipy.stats import skew

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class NormalizerSpaceGenerator(PerColumnSpaceGenerator):
    """Single exclusive normalisation choice per numeric column.

    Replaces the old sequential DISTRIBUTE → SCALE → QUANTILE trio.
    Offering all four transformers as independent per-column choices was
    incoherent: QuantileTransformer remaps data by percentile rank and
    destroys any prior distribution fix or standardisation.  A single
    categorical choice per column lets Optuna learn which normalisation
    path (if any) is best for each feature without sampling invalid
    combinations.
    """

    prefix = "norm"
    SKEWNESS_THRESHOLD = 0.5
    QUANTILE_RANGE_OPTIONS = [25.0]

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        options: list[str] = []

        # NonLinear (Yeo-Johnson): only offered for skewed columns.
        col_skew = float(skew(X[col].dropna()))
        if abs(col_skew) >= self.SKEWNESS_THRESHOLD:
            options.append("nonlinear__yeo_johnson")

        # Normalization (L2 row-wise).
        options.append("normalize__l2")

        # Standard and robust scaling.
        options.append("scale__standard")
        for q_low in self.QUANTILE_RANGE_OPTIONS:
            options.append(f"scale__robust__{q_low}")

        # Quantile transform (uniform and normal output distributions).
        options.extend(["quantile__uniform", "quantile__normal"])

        return options
