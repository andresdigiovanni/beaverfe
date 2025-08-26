import random

import numpy as np
from sklearn.feature_selection import RFECV, mutual_info_classif, mutual_info_regression

from beaverfe.auto_parameters.shared import PermutationRFECV
from beaverfe.transformations import MathematicalOperations
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class MathematicalOperationsParameterSelector:
    SYMMETRIC_OPERATIONS = ["add", "subtract", "multiply"]
    NON_SYMMETRIC_OPERATIONS = ["divide"]
    MAX_ROUNDS = 5

    def select_best_parameters(
        self, x, y, model, scoring, direction, cv, groups, tol, logger: VerboseLogger
    ):
        logger.task_start("Starting mathematical operations search")

        numeric_columns = dtypes.numerical_columns(x)
        n_numeric = len(numeric_columns)
        if n_numeric == 0:
            logger.warn("No numerical columns found for mathematical operations.")
            return None

        # --- Adjust dinamically the hyperparameters ---
        block_size = min(200, max(20, 4 * n_numeric))
        top_k = min(30, max(10, int(0.25 * n_numeric)))

        transformations_map, operation_candidates = self._generate_operations(
            x, numeric_columns
        )

        # --- Multiples iterations ---
        candidate_features = None
        prev_selected = set()

        for round_idx in range(1, self.MAX_ROUNDS + 1):
            if candidate_features is None:
                ops_to_eval = operation_candidates
            else:
                ops_to_eval = [
                    transformations_map[col]
                    for col in candidate_features
                    if col in transformations_map
                ]

            if not ops_to_eval:
                break

            random.shuffle(ops_to_eval)
            blocks = self._split_into_blocks(ops_to_eval, block_size)
            n_blocks = len(blocks)

            # --- Evaluate block ---
            candidate_features = []
            for i, block in enumerate(blocks, start=1):
                transformer = MathematicalOperations(block)
                x_block = transformer.fit_transform(x)

                selected = self._quick_filter(x_block, y, top_k)
                candidate_features.extend(selected)
                logger.progress(
                    f"   ↪ Evaluating round {round_idx}, block {i}/{n_blocks}"
                )

            if not candidate_features:
                break

            logger.task_update(
                f"Round {round_idx}: {len(candidate_features)} features selected"
            )

            # condición de parada
            if len(candidate_features) <= top_k:
                break
            if set(candidate_features) == prev_selected:
                logger.task_update("Convergence reached (no change across rounds)")
                break

            prev_selected = set(candidate_features)

        # --- Refine with RFECV ---
        if not candidate_features:
            logger.warn("No mathematical operations were selected")
            return None

        logger.task_update("Final selection")

        selected_ops = [
            transformations_map[col]
            for col in candidate_features
            if col in transformations_map
        ]
        final_ops = self._select_final_columns(
            x,
            y,
            model,
            scoring,
            cv,
            groups,
            selected_ops,
            transformations_map,
            logger,
        )

        if not final_ops:
            logger.warn("No mathematical operations passed the final refinement")
            return None

        logger.task_result(f"Selected {len(final_ops)} mathematical operation(s)")
        transformer = MathematicalOperations(final_ops)
        return {
            "name": transformer.__class__.__name__,
            "params": transformer.get_params(),
        }

    # --- Métodos auxiliares (idénticos a los anteriores) ---
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

    def _quick_filter(self, X_block, y, top_k):
        try:
            if len(np.unique(y)) < 20:
                scores = mutual_info_classif(X_block, y, discrete_features="auto")
            else:
                scores = mutual_info_regression(X_block, y, discrete_features="auto")

            ranking = np.argsort(scores)[::-1]
            selected = [X_block.columns[i] for i in ranking[:top_k] if scores[i] > 0]
            return selected

        except Exception:
            return list(X_block.columns)

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
