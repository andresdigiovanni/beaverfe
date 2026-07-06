import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator
from beaverfe.model_profiler import DataProfiler
from beaverfe.transformations.utils import dtypes


class MissingValuesHandlerSpaceGenerator(SpaceGenerator):
    DEFAULT_KNN_NEIGHBORS = [5]
    # The k-NN based mutual-information estimator has finite-sample bias, so
    # even genuinely independent columns rarely score exactly 0 — a small
    # nonzero floor is needed to distinguish real signal from that noise.
    MUTUAL_INFO_THRESHOLD = 0.05

    def __init__(self) -> None:
        self._profiler: DataProfiler | None = None

    def set_profiler(self, profiler: DataProfiler) -> None:
        """Inject a :class:`~beaverfe.model_profiler.DataProfiler` to enable
        data-aware search space filtering.

        Args:
            profiler: Frozen profile computed from the training data.
        """
        self._profiler = profiler

    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        cat_cols = dtypes.categorical_columns(X)
        num_cols = dtypes.numerical_columns(X)
        X_filtered = X[cat_cols + num_cols]

        missing_cols = self._columns_with_nulls(X_filtered)
        if not missing_cols:
            return {}

        # KNN imputation is O(n·k) per row at transform time — prohibitively
        # slow at scale.  Skip it entirely when the dataset is large.
        skip_knn = self._profiler is not None and self._profiler.is_large_sample

        space: SearchSpace = {}
        for col in missing_cols:
            is_numeric = col in num_cols
            strategies = self._candidate_strategies(X_filtered[col], y, is_numeric)
            # Encode knn with neighbors folded into key: "knn_5"
            encoded: list[str] = []
            for strategy, params in strategies.items():
                if strategy == "knn":
                    if skip_knn:
                        continue
                    for n in params.get("n_neighbors", self.DEFAULT_KNN_NEIGHBORS):
                        encoded.append(f"knn_{n}")
                else:
                    encoded.append(strategy)
            if encoded:
                space[f"impute_{col}"] = encoded

        return space

    def _columns_with_nulls(self, X: pd.DataFrame) -> list[str]:
        return list(X.columns[X.isnull().any()])

    def _candidate_strategies(
        self,
        X_col: pd.Series,
        y: pd.Series | np.ndarray,
        is_numeric: bool,
    ) -> dict[str, dict]:
        strategies: dict[str, dict] = {}

        if is_numeric:
            std = X_col.std()
            skew_val = X_col.skew()

            if abs(skew_val) > 1:
                strategies["median"] = {}
            else:
                strategies["mean"] = {}

            if std < 1 or (X_col == 0).sum() / len(X_col) > 0.3:
                strategies["fill_0"] = {}

            if self._is_column_important(X_col, y):
                strategies["knn"] = {"n_neighbors": self.DEFAULT_KNN_NEIGHBORS}
        else:
            strategies["fill_0"] = {}

            if X_col.nunique() <= 15:
                strategies["most_frequent"] = {}

        return strategies

    def _is_column_important(self, X_col: pd.Series, y: pd.Series | np.ndarray) -> bool:
        # LOOK-AHEAD NOTE (menu-curation bias, not label leakage into the shipped model):
        # Correlation and mutual information here are computed against the full y,
        # including future CV validation rows. This determines only whether KNN impute
        # is *offered* in the search space — it does not affect the imputed values
        # used during model evaluation (those are fit inside CV folds). Severity: LOW.
        if X_col.dtype.kind not in "bifc":
            return False

        y_arr = np.asarray(y)
        valid = X_col.notna().to_numpy()
        if valid.sum() < 2:
            return False

        x_arr = X_col.to_numpy()[valid]
        y_valid = y_arr[valid]
        n_unique_y = len(np.unique(y_valid))

        if 2 < n_unique_y < 20:
            # Multiclass-like target: Pearson correlation assumes an ordinal
            # relationship that arbitrary class labels don't have. Mutual
            # information handles a nominal target correctly instead.
            try:
                score = mutual_info_classif(
                    x_arr.reshape(-1, 1),
                    y_valid,
                    discrete_features=False,
                    random_state=42,
                )[0]
                return bool(score > self.MUTUAL_INFO_THRESHOLD)
            except ValueError:
                return False

        if y_arr.dtype.kind in "bifc":
            with np.errstate(invalid="ignore"):
                correlation = np.corrcoef(x_arr, y_valid)[0, 1]
            return bool(not np.isnan(correlation) and abs(correlation) > 0.3)
        return False
