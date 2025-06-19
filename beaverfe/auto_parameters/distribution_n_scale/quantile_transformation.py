from typing import Any, Dict, Optional

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import QuantileTransformation
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class QuantileTransformationParameterSelector:
    TRANSFORMATION_OPTIONS = ["uniform", "normal"]

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
    ) -> Optional[Dict[str, Any]]:
        logger.task_start("Starting search for optimal quantile transformations.")

        columns = dtypes.numerical_columns(X)
        if not columns:
            logger.warn("No numerical columns found for quantile transformations.")
            return None

        n_columns = len(columns)
        selected_transformations = {}

        base_score = evaluate_model(X, y, model, scoring, cv, groups)
        logger.baseline(
            f"Baseline score (no quantile transformation): {base_score:.4f}"
        )

        for index, column in enumerate(columns, start=1):
            logger.task_update(f"[{index}/{n_columns}] Evaluating column: '{column}'")

            best_option = self._evaluate_column_transformations(
                X,
                y,
                model,
                scoring,
                base_score,
                column,
                direction,
                cv,
                groups,
                tol,
                logger,
            )

            if best_option:
                selected_transformations.update(best_option)
                logger.task_result(
                    f"Selected transformation for '{column}': {list(best_option.values())[0]}"
                )

        if selected_transformations:
            logger.task_result(
                f"Quantile transformation selected for {len(selected_transformations)} column(s)."
            )
            return self._build_transformation_result(selected_transformations)

        logger.warn("No quantile transformation improved performance.")
        return None

    def _evaluate_column_transformations(
        self,
        X,
        y,
        model,
        scoring,
        base_score: float,
        column: str,
        direction: str,
        cv,
        groups,
        tol,
        logger: VerboseLogger,
    ) -> Dict[str, str]:
        best_score = base_score
        best_transformation = {}

        for option in self.TRANSFORMATION_OPTIONS:
            transformation = QuantileTransformation({column: option})
            score = evaluate_model(X, y, model, scoring, cv, groups, transformation)
            logger.progress(f"   ↪ Tried '{option}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction, tol):
                best_score = score
                best_transformation = {column: option}

        return best_transformation

    def _build_transformation_result(
        self, transformation_options: Dict[str, str]
    ) -> Dict[str, Any]:
        transformer = QuantileTransformation(
            transformation_options=transformation_options
        )
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
