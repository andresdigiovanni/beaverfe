from typing import Any, Dict, Optional

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import ScaleTransformation
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class ScaleTransformationParameterSelector:
    SCALER_OPTIONS = ["standard", "robust"]
    QUANTILE_RANGE_OPTIONS = [25.0]

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
        logger.task_start("Starting search for optimal scaling transformations.")

        columns = dtypes.numerical_columns(X)
        if not columns:
            logger.warn("No numerical columns found for scaling transformations.")
            return None

        n_columns = len(columns)
        selected_scalers = {"transformation_options": {}, "quantile_range": {}}

        base_score = evaluate_model(X, y, model, scoring, cv, groups)
        logger.baseline(f"Baseline score (no scaling): {base_score:.4f}")

        for index, column in enumerate(columns, start=1):
            logger.task_update(f"[{index}/{n_columns}] Evaluating column: '{column}'")

            best_for_column = self._evaluate_scalers_for_column(
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

            if best_for_column:
                selected_scalers["transformation_options"].update(
                    best_for_column.get("transformation_options", {})
                )
                selected_scalers["quantile_range"].update(
                    best_for_column.get("quantile_range", {})
                )

                scaler = selected_scalers["transformation_options"][column]
                if column in selected_scalers["quantile_range"]:
                    q_range = selected_scalers["quantile_range"][column]
                    logger.task_result(
                        f"Selected scaler for '{column}': {scaler} (quantile range {q_range})"
                    )
                else:
                    logger.task_result(f"Selected scaler for '{column}': {scaler}")

        if selected_scalers["transformation_options"]:
            logger.task_result(
                f"Scaling applied to {len(selected_scalers['transformation_options'])} column(s)."
            )
            return self._build_transformation_result(selected_scalers)

        logger.warn("No scaling transformation improved performance.")
        return None

    def _evaluate_scalers_for_column(
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
    ) -> Dict[str, Dict[str, Any]]:
        best_score = base_score
        best_params = {}

        for scaler in self.SCALER_OPTIONS:
            if scaler == "robust":
                for q_low in self.QUANTILE_RANGE_OPTIONS:
                    q_range = (q_low, 100.0 - q_low)
                    params = {
                        "transformation_options": {column: scaler},
                        "quantile_range": {column: q_range},
                    }
                    transformer = ScaleTransformation(**params)
                    score = evaluate_model(
                        X, y, model, scoring, cv, groups, transformer
                    )
                    logger.progress(
                        f"   ↪ Tried '{scaler}' (quantile range {q_range}) → Score: {score:.4f}"
                    )

                    if is_score_improved(score, best_score, direction, tol):
                        best_score = score
                        best_params = params
            else:
                params = {"transformation_options": {column: scaler}}
                transformer = ScaleTransformation(**params)
                score = evaluate_model(X, y, model, scoring, cv, groups, transformer)
                logger.progress(f"   ↪ Tried '{scaler}' → Score: {score:.4f}")

                if is_score_improved(score, best_score, direction, tol):
                    best_score = score
                    best_params = params

        return best_params

    def _build_transformation_result(
        self, selected_scalers: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        transformer = ScaleTransformation(**selected_scalers)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
