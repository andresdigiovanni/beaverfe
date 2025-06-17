from typing import Any, Dict

from beaverfe.auto_parameters.shared import RecursiveFeatureAddition
from beaverfe.transformations import ColumnSelection
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class ColumnSelectionParameterSelector:
    def select_best_parameters(
        self, X, y, model, scoring, direction: str, cv, groups, logger: VerboseLogger
    ) -> Dict[str, Any]:
        """
        Selects the most informative subset of features using recursive feature addition.
        """

        numeric_columns = dtypes.numerical_columns(X)
        X_filtered = X[numeric_columns]

        logger.task_start("Beginning feature selection")

        selector = RecursiveFeatureAddition(
            model, scoring, direction, cv, groups, early_stopping=5, logger=logger
        )
        selected_features = selector.fit(X_filtered, y)

        logger.task_result(f"{len(selected_features)} feature(s) selected")

        transformer = ColumnSelection(selected_features)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
