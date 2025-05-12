from collections import ChainMap
from typing import Dict, Optional, Tuple

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import MissingValuesHandler
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class MissingValuesHandlerParameterSelector:
    def __init__(self):
        self.STRATEGIES = {
            "all": {
                "fill_0": {},
                "most_frequent": {},
            },
            "num": {
                "mean": {},
                "median": {},
                "knn": {"n_neighbors": [5]},
            },
            "cat": {},
        }

    def select_best_parameters(
        self, X, y, model, scoring, direction: str, cv, groups, logger: VerboseLogger
    ) -> Optional[Dict[str, object]]:
        cat_columns = dtypes.categorical_columns(X)
        num_columns = dtypes.numerical_columns(X)
        X = X[cat_columns + num_columns]

        logger.task_start("Starting missing value imputation optimization")

        missing_columns = self._columns_with_nulls(X)
        total_columns = len(missing_columns)

        if not total_columns:
            logger.warn("No missing values found. Skipping imputation transformation.")
            return None

        best_strategies = {}
        best_knn_params = {}

        for i, column in enumerate(missing_columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")
            is_numeric = column in num_columns

            strategy, params = self._find_best_strategy_for_column(
                X, y, model, scoring, direction, cv, groups, column, logger, is_numeric
            )

            best_strategies[column] = strategy
            logger.task_result(f"Selected imputation for '{column}': {strategy}")

            if strategy == "knn":
                best_knn_params.update(params)

        logger.task_result(f"Imputation applied to {len(best_strategies)} column(s)")

        return self._build_result(best_strategies, best_knn_params)

    def _columns_with_nulls(self, X):
        return X.columns[X.isnull().any()].tolist()

    def _find_best_strategy_for_column(
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
        base_score = float("-inf") if direction == "maximize" else float("inf")
        best_strategy = None
        best_params = {}

        strategies = ChainMap(
            self.STRATEGIES["all"],
            self.STRATEGIES["num"] if is_numeric else self.STRATEGIES["cat"],
        )

        for strategy, params in strategies.items():
            if strategy == "knn":
                score, knn_param = self._evaluate_knn(
                    X, y, model, scoring, direction, cv, groups, column, params, logger
                )
            else:
                score = self._evaluate_strategy(
                    X, y, model, scoring, cv, groups, column, strategy, logger
                )
                knn_param = {}
                logger.progress(f"   ↪ Tried '{strategy}' → Score: {score:.4f}")

            if is_score_improved(score, base_score, direction):
                base_score = score
                best_strategy = strategy
                best_params = knn_param

        return best_strategy, best_params

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
        logger: VerboseLogger,
    ) -> Tuple[float, Dict[str, int]]:
        best_score = float("-inf") if direction == "maximize" else float("inf")
        best_param = {}

        for n_neighbors in params.get("n_neighbors", []):
            score = self._evaluate_strategy(
                X, y, model, scoring, cv, groups, column, "knn", logger, n_neighbors
            )
            logger.progress(
                f"   ↪ Tried 'knn' with n_neighbors={n_neighbors} → Score: {score:.4f}"
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
        logger: VerboseLogger,
        n_neighbors: Optional[int] = None,
    ) -> float:
        transformation_options = {column: strategy}
        knn_params = {column: n_neighbors} if n_neighbors is not None else None

        transformer = MissingValuesHandler(
            transformation_options=transformation_options, n_neighbors=knn_params
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
