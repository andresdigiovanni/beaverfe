import random

from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

from beaverfe.auto_parameters.shared import PermutationRFECV
from beaverfe.transformations import MathematicalOperations
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class MathematicalOperationsParameterSelector:
    SYMMETRIC_OPERATIONS = ["add", "subtract", "multiply"]
    NON_SYMMETRIC_OPERATIONS = ["divide"]
    BLOCK_SIZE = 20

    def select_best_parameters(
        self, x, y, model, scoring, direction, cv, groups, tol, logger: VerboseLogger
    ):
        logger.task_start("Starting mathematical operations search")

        numeric_columns = dtypes.numerical_columns(x)
        if not numeric_columns:
            logger.warn("No numerical columns found for mathematical operations.")
            return None

        transformations_map, operation_candidates = self._generate_operations(
            x, numeric_columns
        )
        random.shuffle(operation_candidates)
        blocks = self._split_into_blocks(operation_candidates, self.BLOCK_SIZE)

        selected_operations = self._evaluate_blocks(
            x, y, model, direction, tol, blocks, transformations_map, logger
        )

        if not selected_operations:
            logger.warn("No mathematical operations were selected")
            return None

        logger.task_update("Selecting best operations")

        final_operations = self._select_final_columns(
            x,
            y,
            model,
            scoring,
            cv,
            groups,
            selected_operations,
            transformations_map,
            logger,
        )

        if not final_operations:
            logger.warn("No mathematical operations passed the final refinement")
            return None

        logger.task_result(
            f"Selected {len(final_operations)} mathematical operation(s)"
        )
        transformer = MathematicalOperations(final_operations)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }

    def _generate_operations(self, x, columns):
        transformations = {}
        operations = []

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    continue

                for op in self._operation_definitions(i, j):
                    op_tuple = (col1, col2, op)
                    operations.append(op_tuple)

                    transformed_col = self._apply_transformation_and_get_column(
                        x, op_tuple
                    )
                    transformations[transformed_col] = op_tuple

        return transformations, operations

    def _operation_definitions(self, i, j):
        definitions = []
        for op in self.SYMMETRIC_OPERATIONS:
            if i > j:
                definitions.append(op)

        for op in self.NON_SYMMETRIC_OPERATIONS:
            definitions.append(op)

        return definitions

    def _apply_transformation_and_get_column(self, x, op_tuple):
        transformer = MathematicalOperations([op_tuple])
        transformed = transformer.fit_transform(x)
        return next(col for col in transformed.columns if col not in x.columns)

    def _split_into_blocks(self, items, block_size):
        return [items[i : i + block_size] for i in range(0, len(items), block_size)]

    def _evaluate_blocks(
        self, x, y, model, direction, tol, blocks, transformations_map, logger
    ):
        selected_ops = []

        for i, block in enumerate(blocks, start=1):
            logger.task_update(f"Evaluating block {i}/{len(blocks)}")

            transformer = MathematicalOperations(block)
            x_transformed = transformer.fit_transform(x)

            model_clone = clone(model)
            model_clone.fit(x_transformed, y)

            importances = permutation_importance(
                model_clone, x_transformed, y, n_repeats=5
            )
            feature_scores = dict(
                zip(x_transformed.columns, importances.importances_mean)
            )

            selected_in_block = self._select_features_from_block(
                feature_scores, transformations_map, direction, tol
            )

            selected_ops.extend(selected_in_block)
            logger.progress(
                f"   ↪ Block {i}: {len(selected_in_block)} selected features"
            )

        return selected_ops

    def _select_features_from_block(self, scores, transformations_map, direction, tol):
        selected = []

        for new_col, new_score in scores.items():
            if new_col not in transformations_map:
                continue

            op_tuple = transformations_map[new_col]
            col1, col2, _ = op_tuple

            if direction == "maximize":
                base_score = max(
                    scores.get(col1, float("-inf")), scores.get(col2, float("-inf"))
                )
            else:
                base_score = min(
                    scores.get(col1, float("inf")), scores.get(col2, float("inf"))
                )

            score_improved = (
                direction == "maximize" and new_score > base_score + tol
            ) or (direction == "minimize" and new_score < base_score - tol)

            if score_improved:
                selected.append(op_tuple)

        return selected

    def _select_final_columns(
        self, x, y, model, scoring, cv, groups, operations, transformations_map, logger
    ):
        transformer = MathematicalOperations(operations)
        x_transformed = transformer.fit_transform(x, y)

        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            rfecv = RFECV(estimator=model, scoring=scoring, cv=cv, step=0.2)
        else:
            rfecv = PermutationRFECV(estimator=model, scoring=scoring, cv=cv, step=0.2)

        rfecv.fit(x_transformed, y, groups=groups)
        selected_columns = list(rfecv.get_feature_names_out())

        return [
            transformations_map[col]
            for col in selected_columns
            if col in transformations_map
        ]
