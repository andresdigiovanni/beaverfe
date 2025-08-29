from sklearn.feature_selection import RFECV

from beaverfe.auto_parameters.shared import PermutationRFECV
from beaverfe.transformations import CyclicalFeaturesTransformer
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class CyclicalFeaturesTransformerParameterSelector:
    VALID_SUFFIX_PERIODS = {
        "month": 12,
        "day": 31,
        "weekday": 7,
        "hour": 24,
        "minute": 60,
        "second": 60,
    }

    def select_best_parameters(
        self, x, y, model, scoring, direction, cv, groups, tol, logger: VerboseLogger
    ):
        logger.task_start("Detecting cyclical features")

        columns = dtypes.numerical_columns(x)
        if not columns:
            logger.warn("No numerical columns found for cyclical features.")
            return None

        transformations = {}

        for column in columns:
            period = self._infer_cyclical_period(column)
            if period:
                transformations[column] = period

        transformations = self._select_final_columns(
            x, y, model, scoring, cv, groups, transformations
        )

        if transformations:
            logger.task_result(
                f"Cyclical features applied to {len(transformations)} column(s)"
            )
            return self._build_transformation_result(transformations)

        logger.warn("No cyclical features were applied to any column")
        return None

    def _infer_cyclical_period(self, column_name):
        """Infer the cyclical period of a column based on its name or unique value count."""
        column_name_lower = column_name.lower()

        # Check suffix match for common time units
        for suffix, period in self.VALID_SUFFIX_PERIODS.items():
            if column_name_lower.endswith(suffix):
                return period

        return None

    def _select_final_columns(self, x, y, model, scoring, cv, groups, transformations):
        transformer = CyclicalFeaturesTransformer(transformations)
        x_transformed = transformer.fit_transform(x, y)

        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            rfecv = RFECV(estimator=model, scoring=scoring, cv=cv, step=0.1)
        else:
            rfecv = PermutationRFECV(estimator=model, scoring=scoring, cv=cv, step=0.1)

        rfecv.fit(x_transformed, y, groups=groups)
        selected_columns = list(rfecv.get_feature_names_out())

        return {k: v for k, v in transformations.items() if k in selected_columns}

    def _build_transformation_result(self, transformation_options):
        transformer = CyclicalFeaturesTransformer(transformation_options)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
