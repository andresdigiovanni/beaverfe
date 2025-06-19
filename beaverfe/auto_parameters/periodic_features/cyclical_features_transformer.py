from beaverfe.transformations import CyclicalFeaturesTransformer
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class CyclicalFeaturesTransformerParameterSelector:
    VALID_SUFFIX_PERIODS = {
        "_month": 12,
        "_day": 31,
        "_weekday": 7,
        "_hour": 24,
        "_minute": 60,
        "_second": 60,
    }

    UNIQUE_VALUE_RATIO_THRESHOLD = 0.10

    def select_best_parameters(
        self, x, y, model, scoring, direction, cv, groups, tol, logger: VerboseLogger
    ):
        logger.task_start("Detecting cyclical features")

        numerical_columns = dtypes.numerical_columns(x)
        transformation_options = {}

        for column in numerical_columns:
            period = self._infer_cyclical_period(x, column)
            if period:
                transformation_options[column] = period

        if transformation_options:
            logger.task_result(
                f"Cyclical features applied to {len(transformation_options)} column(s)"
            )
            return self._build_transformation_result(transformation_options)

        logger.warn("No cyclical features were applied to any column")
        return None

    def _infer_cyclical_period(self, dataframe, column_name):
        """Infer the cyclical period of a column based on its name or unique value count."""
        column_name_lower = column_name.lower()

        # Check suffix match for common time units
        for suffix, period in self.VALID_SUFFIX_PERIODS.items():
            if column_name_lower.endswith(suffix):
                return period

        # Fallback: check if column has low unique value ratio
        unique_values = dataframe[column_name].dropna().unique()
        unique_ratio = len(unique_values) / len(dataframe)

        if len(unique_values) > 2 and unique_ratio < self.UNIQUE_VALUE_RATIO_THRESHOLD:
            return len(unique_values)

        return None

    def _build_transformation_result(self, transformation_options):
        transformer = CyclicalFeaturesTransformer(transformation_options)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
