from typing import Any, Dict, Optional

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import CategoricalEncoding
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class CategoricalEncodingParameterSelector:
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
        logger.task_start("Starting search for optimal categorical encodings.")

        is_binary_target = y.nunique() == 2
        candidate_encodings = self._get_candidate_encodings(X, is_binary_target)
        total_columns = len(candidate_encodings)
        best_encoding_config = {}

        for index, (column, encodings) in enumerate(
            candidate_encodings.items(), start=1
        ):
            logger.task_update(
                f"[{index}/{total_columns}] Evaluating encodings for column: '{column}'"
            )

            column, optimal_encoding = self._evaluate_column(
                X, y, model, scoring, direction, cv, groups, logger, column, encodings
            )

            if optimal_encoding:
                logger.task_result(
                    f"Selected encoding for '{column}': {optimal_encoding}"
                )
                best_encoding_config[column] = optimal_encoding

        if best_encoding_config:
            logger.task_result(
                f"Encoding strategy selected for {len(best_encoding_config)} column(s)."
            )
            final_encoder = CategoricalEncoding(best_encoding_config)
            return {
                "name": final_encoder.__class__.__name__,
                "params": final_encoder.get_params(),
            }

        logger.warn("No suitable categorical encodings were identified.")
        return None

    def _get_candidate_encodings(self, X, is_binary_target) -> Dict[str, list]:
        categorical_columns = dtypes.categorical_columns(X)
        column_category_counts = {col: X[col].nunique() for col in categorical_columns}

        encoding_options = {}

        for column, n_categories in column_category_counts.items():
            encoders = []

            if n_categories == 2:
                encoders = ["dummy"]

            elif n_categories <= 15:
                encoders = ["dummy", "catboost", "target"]
                if is_binary_target:
                    encoders.append("woe")

            elif n_categories <= 50:
                encoders = ["catboost", "binary", "target", "loo"]

            else:
                encoders = ["catboost", "hashing", "target"]

            encoding_options[column] = encoders

        return encoding_options

    def _evaluate_column(
        self, X, y, model, scoring, direction, cv, groups, logger, column, encodings
    ):
        if len(encodings) == 1:
            return column, encodings[0]

        best_score = float("-inf") if direction == "maximize" else float("inf")
        optimal_encoding = None

        for encoding_method in encodings:
            current_encoding = {column: encoding_method}
            transformer = CategoricalEncoding(current_encoding)

            score = evaluate_model(X, y, model, scoring, cv, groups, transformer)
            logger.progress(f"   ↪ Tried '{encoding_method}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction):
                best_score = score
                optimal_encoding = encoding_method

        return column, optimal_encoding
