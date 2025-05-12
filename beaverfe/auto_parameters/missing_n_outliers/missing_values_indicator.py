from beaverfe.transformations import MissingValuesIndicator
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class MissingValuesIndicatorParameterSelector:
    def select_best_parameters(
        self, X, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        # Keep only categorical and numerical columns
        cat_columns = dtypes.categorical_columns(X)
        num_columns = dtypes.numerical_columns(X)
        X = X[cat_columns + num_columns]

        logger.task_start("Starting missing value indicator evaluation")

        columns_with_nulls = self._get_columns_with_nulls(X)

        if not columns_with_nulls:
            logger.warn("No missing values found. Skipping indicator transformation.")
            return None

        logger.task_result(
            f"Selected {len(columns_with_nulls)} column(s) with missing values"
        )

        return self._build_result(columns_with_nulls)

    def _get_columns_with_nulls(self, X):
        return [col for col in X.columns if X[col].isnull().any()]

    def _build_result(self, features):
        indicator = MissingValuesIndicator(features=features)
        return {
            "name": indicator.__class__.__name__,
            "params": indicator.get_params(),
        }
