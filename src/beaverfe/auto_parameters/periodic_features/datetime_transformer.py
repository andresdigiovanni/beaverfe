import numpy as np
import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator
from beaverfe.transformations.utils import dtypes


class DateTimeTransformerSpaceGenerator(SpaceGenerator):
    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        columns = dtypes.datetime_columns(X)
        if not columns:
            return {}
        return {"datetime_columns": [columns]}
