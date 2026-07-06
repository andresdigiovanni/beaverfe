import numpy as np
import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator
from beaverfe.transformations.utils import dtypes


class ColumnSelectionSpaceGenerator(SpaceGenerator):
    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        columns = dtypes.numerical_columns(X)
        if not columns:
            return {}
        # Column selection has no dedicated pipeline block and is never
        # sampled via Optuna; this candidate list is only consumed by
        # PipelineAssembler._decode_dim_reduction. The actual
        # column-elimination decision happens post-hoc, via permutation
        # importance, in auto_feature_pipeline._prune_eliminated_columns.
        return {"column_selection_candidates": [columns]}
