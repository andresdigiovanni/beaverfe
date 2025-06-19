from typing import Dict, Optional, Tuple

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import MissingValuesHandler
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class MissingValuesHandlerParameterSelector:
    DEFAULT_KNN_NEIGHBORS = [5]

    def select_best_parameters(
        self,
        X,
        y,
        model,
        scoring,
        direction: str,
        cv,
        groups,
        tol,
        logger: VerboseLogger,
    ) -> Optional[Dict[str, object]]:
        cat_cols = dtypes.categorical_columns(X)
        num_cols = dtypes.numerical_columns(X)
        X = X[cat_cols + num_cols]

        logger.task_start("Starting missing value imputation optimization")

        missing_cols = self._columns_with_nulls(X)
        if not missing_cols:
            logger.warn("No missing values found. Skipping imputation transformation.")
            return None

        best_strategies = {}
        best_knn_params = {}

        for i, col in enumerate(missing_cols, start=1):
            logger.task_update(f"[{i}/{len(missing_cols)}] Evaluating column: '{col}'")
            is_numeric = col in num_cols

            strategy, knn_param = self._select_column_strategy(
                X, y, model, scoring, direction, cv, groups, col, logger, is_numeric
            )

            best_strategies[col] = strategy
            if strategy == "knn":
                best_knn_params.update(knn_param)

            logger.task_result(f"Selected imputation for '{col}': {strategy}")

        logger.task_result(f"Imputation applied to {len(best_strategies)} column(s)")

        return self._build_result(best_strategies, best_knn_params)

    def _columns_with_nulls(self, X):
        return X.columns[X.isnull().any()].tolist()

    def _select_column_strategy(
        self,
        X,
        y,
        model,
        scoring,
        direction: str,
        cv,
        groups,
        column: str,
        logger: VerboseLogger,
        is_numeric: bool,
    ) -> Tuple[str, Dict[str, int]]:
        candidate_strategies = self._candidate_strategies(X[column], y, is_numeric)

        if len(candidate_strategies) == 1:
            strategy = next(iter(candidate_strategies))
            logger.progress(
                f"   ↪ Only one strategy '{strategy}' — selected without evaluation."
            )

            if strategy == "knn":
                return strategy, {
                    column: candidate_strategies[strategy]["n_neighbors"][0]
                }

            return strategy, {}

        best_score = float("-inf") if direction == "maximize" else float("inf")
        best_strategy = None
        best_params = {}

        for strategy, params in candidate_strategies.items():
            if strategy == "knn":
                score, knn_param = self._evaluate_knn(
                    X, y, model, scoring, direction, cv, groups, column, params
                )
            else:
                score = self._evaluate_strategy(
                    X, y, model, scoring, cv, groups, column, strategy
                )
                knn_param = {}

            logger.progress(f"   ↪ Tried '{strategy}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_strategy = strategy
                best_params = knn_param

        return best_strategy, best_params

    def _candidate_strategies(self, X_col, y, is_numeric: bool) -> Dict[str, dict]:
        strategies = {}

        if is_numeric:
            std = X_col.std()
            skew = X_col.skew()

            if abs(skew) > 1:
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

    def _evaluate_knn(
        self,
        X,
        y,
        model,
        scoring,
        direction: str,
        cv,
        groups,
        column: str,
        params: Dict[str, list],
    ) -> Tuple[float, Dict[str, int]]:
        best_score = float("-inf") if direction == "maximize" else float("inf")
        best_param = {}

        for n_neighbors in params.get("n_neighbors", []):
            score = self._evaluate_strategy(
                X, y, model, scoring, cv, groups, column, "knn", n_neighbors
            )

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_param = {column: n_neighbors}

        return best_score, best_param

    def _evaluate_strategy(
        self,
        X,
        y,
        model,
        scoring,
        cv,
        groups,
        column: str,
        strategy: str,
        n_neighbors: Optional[int] = None,
    ) -> float:
        transformer = MissingValuesHandler(
            transformation_options={column: strategy},
            n_neighbors={column: n_neighbors} if n_neighbors is not None else None,
        )
        return evaluate_model(X, y, model, scoring, cv, groups, transformer)

    def _build_result(
        self, strategies: Dict[str, str], knn_params: Dict[str, int]
    ) -> Dict[str, object]:
        transformer = MissingValuesHandler(
            transformation_options=strategies, n_neighbors=knn_params
        )
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }

    def _is_column_important(self, X_col, y) -> bool:
        if X_col.dtype.kind in "bifc" and y.dtype.kind in "bifc":
            return abs(X_col.corr(y)) > 0.3

        return False
