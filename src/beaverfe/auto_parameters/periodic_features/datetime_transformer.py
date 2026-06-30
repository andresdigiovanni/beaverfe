from beaverfe.transformations import DateTimeTransformer
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class DateTimeTransformerParameterSelector:
    def select_best_parameters(
        self, X, y, model, scoring, direction, cv, groups, tol, logger: VerboseLogger
    ):
        logger.task_start("Detecting datetime features")

        columns = dtypes.datetime_columns(X)
        if not columns:
            logger.warn("No datetime transformations were applied to any column")
            return None

        logger.task_result(
            f"Datetime transformations applied to {len(columns)} column(s)"
        )

        transformer = DateTimeTransformer(columns)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
