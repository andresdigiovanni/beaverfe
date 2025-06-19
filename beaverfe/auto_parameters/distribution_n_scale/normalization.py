from typing import Any, Dict, Optional

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import Normalization
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class NormalizationParameterSelector:
    NORMALIZATION_OPTIONS = ["l2"]

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
        logger.task_start("Starting search for optimal normalization parameters.")

        columns = dtypes.numerical_columns(X)
        if not columns:
            logger.warn("No numerical columns found for normalization parameters.")
            return None

        n_columns = len(columns)
        selected_normalizations = {}

        base_score = evaluate_model(X, y, model, scoring, cv, groups)
        logger.baseline(f"Baseline score (no normalization): {base_score:.4f}")

        for index, column in enumerate(columns, start=1):
            logger.task_update(
                f"[{index}/{n_columns}] Evaluating normalization for column: '{column}'"
            )

            best_option = self._evaluate_column_normalizations(
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
                selected_normalizations.update(best_option)
                logger.task_result(
                    f"Selected normalization for '{column}': {list(best_option.values())[0]}"
                )

        if selected_normalizations:
            logger.task_result(
                f"Normalization strategy selected for {len(selected_normalizations)} column(s)."
            )
            return self._build_transformation_result(selected_normalizations)

        logger.warn("No normalization method improved performance.")
        return None

    def _evaluate_column_normalizations(
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
        best_normalization = {}

        for method in self.NORMALIZATION_OPTIONS:
            transformation = Normalization({column: method})
            score = evaluate_model(X, y, model, scoring, cv, groups, transformation)
            logger.progress(f"   ↪ Tried '{method}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction, tol):
                best_score = score
                best_normalization = {column: method}

        return best_normalization

    def _build_transformation_result(
        self, transformation_options: Dict[str, str]
    ) -> Dict[str, Any]:
        normalization = Normalization(transformation_options=transformation_options)
        return {
            "name": normalization.__class__.__name__,
            "params": normalization.get_params(),
        }
