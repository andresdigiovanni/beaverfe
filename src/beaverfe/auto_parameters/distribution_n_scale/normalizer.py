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
    # Two quantile ranges: (25, 75) standard and (10, 90) more aggressive.
    QUANTILE_RANGE_OPTIONS = [10.0, 25.0]

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        options: list[str] = []

        col_values = X[col].dropna()
        col_skew = float(skew(col_values))
        is_skewed = abs(col_skew) >= self.SKEWNESS_THRESHOLD

        if is_skewed:
            is_positive = bool((col_values > 0).all())

            # Yeo-Johnson: works for any skewed column, positive or not.
            options.append("nonlinear__yeo_johnson")

            # log1p: simple and effective for right-skewed positive data.
            if is_positive and col_skew > 0:
                options.append("nonlinear__log")

            # Box-Cox: strictly requires positive data; often outperforms
            # Yeo-Johnson on right-skewed all-positive features.
            if is_positive:
                options.append("nonlinear__box_cox")

        # Normalization (L2 row-wise).
        options.append("normalize__l2")

        # Scaling options: standard, min-max, max-abs and robust variants.
        options.append("scale__standard")
        options.append("scale__min_max")
        options.append("scale__max_abs")
        for q_low in self.QUANTILE_RANGE_OPTIONS:
            options.append(f"scale__robust__{q_low}")

        # Quantile transform (uniform and normal output distributions).
        options.extend(["quantile__uniform", "quantile__normal"])

        return options
