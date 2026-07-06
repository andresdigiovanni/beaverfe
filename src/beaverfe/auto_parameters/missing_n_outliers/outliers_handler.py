from itertools import product

import numpy as np
import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class OutliersSpaceGenerator(PerColumnSpaceGenerator):
    prefix = "outlier"

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        outlier_methods = self._get_outlier_methods()
        actions = self._get_outlier_actions()

        return [
            self._build_token(action, method, param)
            for action, method, param in self._generate_combinations(
                actions, outlier_methods
            )
            if self._has_outliers(X[col], method, param)
        ]

    def _get_outlier_methods(self) -> dict[str, dict]:
        return {
            "iqr": {"thresholds": [1.5]},
            "zscore": {"thresholds": [3.0]},
            "iforest": {"contamination": [0.05]},
        }

    def _get_outlier_actions(self) -> list[str]:
        return ["cap", "median"]

    def _generate_combinations(self, actions: list[str], methods: dict[str, dict]):
        for method, params in methods.items():
            values = next(iter(params.values()))
            valid_actions = ["median"] if method == "iforest" else actions
            for action, value in product(valid_actions, values):
                yield action, method, value

    # LOOK-AHEAD NOTE (menu-curation only, no y involved):
    # Outlier detection here uses the full X to decide whether to *offer* an
    # outlier strategy in the search space. No target variable is consulted, so
    # this is the lowest-severity look-ahead-bias case — it cannot leak label
    # information, only bias which methods are available for Optuna to choose.
    # Severity: LOWEST.
    def _has_outliers(self, data: pd.Series, method: str, param: float) -> bool:
        clean = data.dropna()
        if method == "iqr":
            q1, q3 = np.percentile(clean, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - param * iqr, q3 + param * iqr
            return bool(clean[(clean < lower) | (clean > upper)].count() > 0)
        if method == "zscore":
            mean, std = clean.mean(), clean.std()
            lower, upper = mean - param * std, mean + param * std
            return bool(clean[(clean < lower) | (clean > upper)].count() > 0)
        # iforest: skip fitting a full IsolationForest per column just to
        # gate a menu item. With contamination=`param`, IsolationForest
        # flags roughly that fraction of points as outliers by construction
        # (it thresholds anomaly scores at the `param` quantile), so the
        # menu item is only ever actually infeasible for degenerate columns
        # (constant, or too few samples to partition).
        return bool(clean.nunique() > 1 and len(clean) >= 10)

    def _build_token(self, action: str, method: str, param: float) -> str:
        return f"{action}__{method}__{param}"
