import numpy as np
import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator


class MissingValuesIndicatorSpaceGenerator(SpaceGenerator):
    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        null_cols = self._get_columns_with_nulls(X)
        if not null_cols:
            return {}
        return {"missing_indicator_features": [null_cols]}

    def _get_columns_with_nulls(self, X: pd.DataFrame) -> list[str]:
        return [col for col in X.columns if X[col].isnull().any()]
