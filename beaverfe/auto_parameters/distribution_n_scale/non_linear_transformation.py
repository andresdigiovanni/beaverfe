from typing import Any, Dict, Optional

from scipy.stats import skew

from beaverfe.transformations import NonLinearTransformation
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class NonLinearTransformationParameterSelector:
    SKEWNESS_THRESHOLD = 0.5

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
        logger.task_start("Starting search for optimal non-linear transformations.")

        numeric_columns = dtypes.numerical_columns(X)
        total_columns = len(numeric_columns)
        selected_transformations = {}

        for index, column in enumerate(numeric_columns, start=1):
            logger.task_update(
                f"[{index}/{total_columns}] Evaluating column: '{column}'"
            )

            column_skew = skew(X[column].dropna())
            logger.progress(f"   ↪ Skewness: {column_skew:.4f}")

            if abs(column_skew) >= self.SKEWNESS_THRESHOLD:
                selected_transformations[column] = "yeo_johnson"
                logger.task_result(f"Selected 'yeo_johnson' for '{column}'")

        if selected_transformations:
            logger.task_result(
                f"Non-linear transformation selected for {len(selected_transformations)} column(s)."
            )
            return self._build_transformation_result(selected_transformations)

        logger.warn("No columns met the threshold for non-linear transformation.")
        return None

    def _build_transformation_result(
        self, transformation_options: Dict[str, str]
    ) -> Dict[str, Any]:
        transformer = NonLinearTransformation(
            transformation_options=transformation_options
        )
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
