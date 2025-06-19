from typing import Any, Dict

from sklearn.feature_selection import RFECV

from beaverfe.auto_parameters.shared import PermutationRFECV
from beaverfe.transformations import ColumnSelection
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class ColumnSelectionParameterSelector:
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
    ) -> Dict[str, Any]:
        """
        Selects the most informative subset of features using recursive feature addition.
        """

        logger.task_start("Starting feature selection")

        columns = dtypes.numerical_columns(X)
        if not columns:
            logger.warn("No numerical columns found for feature selection.")
            return None

        X_filtered = X[columns]

        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            rfecv = RFECV(estimator=model, scoring=scoring, cv=cv)
        else:
            rfecv = PermutationRFECV(estimator=model, scoring=scoring, cv=cv)

        rfecv.fit(X_filtered, y, groups=groups)
        selected_features = list(rfecv.get_feature_names_out())

        logger.task_result(f"{len(selected_features)} feature(s) selected")

        transformer = ColumnSelection(selected_features)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
