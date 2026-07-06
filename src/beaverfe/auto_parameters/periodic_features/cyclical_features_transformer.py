import numpy as np
import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator
from beaverfe.transformations.utils import dtypes


class CyclicalFeaturesTransformerSpaceGenerator(SpaceGenerator):
    VALID_SUFFIX_PERIODS: dict[str, int] = {
        "month": 12,
        "day": 31,
        "weekday": 7,
        "hour": 24,
        "minute": 60,
        "second": 60,
    }

    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        # DateTimeTransformer unconditionally extracts a fixed set of parts
        # (_year/_month/_day/_weekday/_hour/_minute/_second) for every
        # datetime column and runs before this block in CANONICAL_ORDER, so
        # those output names are predictable even though they don't exist in
        # the raw input yet.
        candidate_columns = [
            *dtypes.numerical_columns(X),
            *self._predicted_datetime_part_columns(X),
        ]
        if not candidate_columns:
            return {}

        space: SearchSpace = {}
        for col in candidate_columns:
            period = self._infer_cyclical_period(col)
            if period is not None:
                space[f"cyclical_{col}"] = ["none", period]

        return space

    def _predicted_datetime_part_columns(self, X: pd.DataFrame) -> list[str]:
        return [
            f"{col}_{suffix}"
            for col in dtypes.datetime_columns(X)
            for suffix in self.VALID_SUFFIX_PERIODS
        ]

    def _infer_cyclical_period(self, column_name: str) -> int | None:
        column_name_lower = column_name.lower()
        for suffix, period in self.VALID_SUFFIX_PERIODS.items():
            if column_name_lower.endswith(suffix):
                return period
        return None
