from itertools import product

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import SplineTransformation
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class SplineTransformationParameterSelector:
    N_KNOTS_OPTIONS = [5, 10]
    DEGREE_OPTIONS = [3]
    EXTRAPOLATION_OPTIONS = ["linear"]

    def select_best_parameters(
        self, x, y, model, scoring, direction, cv, groups, tol, logger: VerboseLogger
    ):
        logger.task_start("Starting spline transformation search")

        columns = dtypes.numerical_columns(x)
        if not columns:
            logger.warn("No numerical columns found for spline transformations.")
            return None

        base_score = evaluate_model(x, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        total_columns = len(columns)
        best_transformations = {}

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_params = self._find_best_spline_params_for_column(
                x,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                tol,
                base_score,
                column,
                logger,
            )

            if best_params:
                config = best_params[column]
                logger.task_result(
                    f"Selected spline transformation for '{column}': "
                    f"extrapolation='{config['extrapolation']}', degree={config['degree']}, n_knots={config['n_knots']}"
                )
                best_transformations.update(best_params)

        if best_transformations:
            logger.task_result(
                f"Spline transformations applied to {len(best_transformations)} column(s)"
            )
            return self._build_transformation_result(best_transformations)

        logger.warn("No spline transformations were applied to any column")
        return None

    def _find_best_spline_params_for_column(
        self,
        x,
        y,
        model,
        scoring,
        direction,
        cv,
        groups,
        tol,
        base_score,
        column,
        logger,
    ):
        best_score = base_score
        best_params = {}

        for n_knots, degree, extrapolation in product(
            self.N_KNOTS_OPTIONS, self.DEGREE_OPTIONS, self.EXTRAPOLATION_OPTIONS
        ):
            params = {
                column: {
                    "degree": degree,
                    "n_knots": n_knots,
                    "extrapolation": extrapolation,
                }
            }

            transformer = SplineTransformation(params)
            score = evaluate_model(x, y, model, scoring, cv, groups, transformer)
            logger.progress(
                f"   ↪ Tried extrapolation='{extrapolation}', degree={degree}, n_knots={n_knots} → Score: {score:.4f}"
            )

            if is_score_improved(score, best_score, direction, tol):
                best_score = score
                best_params = params

        return best_params

    def _build_transformation_result(self, transformation_options):
        transformer = SplineTransformation(
            transformation_options=transformation_options
        )
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }
