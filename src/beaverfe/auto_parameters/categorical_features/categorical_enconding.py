import numpy as np
import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator
from beaverfe.transformations.utils import dtypes


class CategoricalEncodingSpaceGenerator(SpaceGenerator):
    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)
        is_binary_target = bool(y_series.nunique() == 2)
        candidate_encodings = self._get_candidate_encodings(X, is_binary_target)
        if not candidate_encodings:
            return {}
        return {
            f"cat_enc_{col}": encoders for col, encoders in candidate_encodings.items()
        }

    def _get_candidate_encodings(
        self, X: pd.DataFrame, is_binary_target: bool
    ) -> dict[str, list[str]]:
        categorical_columns = dtypes.categorical_columns(X)
        encoding_options: dict[str, list[str]] = {}

        for col in categorical_columns:
            n_categories = X[col].nunique()
            if n_categories == 2:
                encoders: list[str] = ["dummy"]
            elif n_categories <= 15:
                encoders = ["dummy", "catboost", "target", "count", "james_stein"]
                if is_binary_target:
                    encoders.append("woe")
            elif n_categories <= 50:
                encoders = [
                    "catboost",
                    "binary",
                    "target",
                    "loo",
                    "count",
                    "james_stein",
                ]
            else:
                encoders = ["catboost", "hashing", "target", "count", "james_stein"]

            encoding_options[col] = encoders

        return encoding_options
