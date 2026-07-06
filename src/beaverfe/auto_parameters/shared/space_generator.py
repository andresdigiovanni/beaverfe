from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from beaverfe.transformations.utils import dtypes

SearchSpace = dict[str, list[Any]]


class SpaceGenerator(ABC):
    """Pure interface: analyse data, return candidate options. Never call evaluate_model."""

    @abstractmethod
    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        """Return dict of parameter names to lists of valid choices. Return {} if not applicable."""


class PerColumnSpaceGenerator(SpaceGenerator):
    """Base for generators that offer one "{prefix}_{col}" menu per numeric
    column, always including "none" as the first option. Subclasses only
    need `prefix` and `_options_for_column`; shared by every generator whose
    space is a per-column menu (outliers, binning, splines, scaling,
    quantile/normalization/non-linear distribution fixes).
    """

    prefix: str

    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        columns = dtypes.numerical_columns(X)
        if not columns:
            return {}

        space: SearchSpace = {}
        for col in columns:
            options = self._options_for_column(X, col)
            # None means "omit this column entirely" (e.g. non-skewed
            # columns for NonLinearTransformation); an empty list still gets
            # a menu — just one with no choice beyond "none".
            if options is None:
                continue
            space[f"{self.prefix}_{col}"] = ["none", *options]

        return space

    @abstractmethod
    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str] | None:
        """Return this column's candidate options (without "none"). Return
        None (not an empty list) to omit the column from the search space
        entirely."""
