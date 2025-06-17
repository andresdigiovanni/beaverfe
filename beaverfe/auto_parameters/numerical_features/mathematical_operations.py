import random

from beaverfe.auto_parameters.shared import (
    ProbeFeatureSelector,
    RecursiveFeatureAddition,
)
from beaverfe.transformations import MathematicalOperations
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class MathematicalOperationsParameterSelector:
    SYMMETRIC_OPERATIONS = ["add", "subtract", "multiply"]
    NON_SYMMETRIC_OPERATIONS = ["divide"]
    BLOCK_SIZE = 10

    def select_best_parameters(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        logger.task_start("Starting mathematical operations search")

        numerical_columns = dtypes.numerical_columns(x)
        all_transformations, operation_list = self._generate_all_operations(
            x, numerical_columns
        )

        random.shuffle(operation_list)
        blocks = self._create_blocks(operation_list, self.BLOCK_SIZE)

        selected_operations = self._evaluate_blocks(
            x, y, model, blocks, all_transformations, logger
        )

        if not selected_operations:
            logger.warn("No mathematical operations were selected")
            return None

        logger.task_update(
            "Refining selected operations using Recursive Feature Addition"
        )
        final_operations = self._refine_operations(
            x,
            y,
            model,
            scoring,
            direction,
            cv,
            groups,
            selected_operations,
            all_transformations,
        )

        if not final_operations:
            logger.warn("No mathematical operations passed the final refinement")
            return None

        logger.task_result(
            f"Selected {len(final_operations)} mathematical operation(s)"
        )
        final_transformer = MathematicalOperations(final_operations)
        return {
            "name": final_transformer.__class__.__name__,
            "params": final_transformer.get_params(),
        }

    def _generate_all_operations(self, x, columns):
        transformations = {}
        operations = []

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    continue

                for op_list, symmetric in [
                    (self.SYMMETRIC_OPERATIONS, True),
                    (self.NON_SYMMETRIC_OPERATIONS, False),
                ]:
                    if symmetric and i <= j:
                        continue

                    for op in op_list:
                        op_tuple = (col1, col2, op)
                        operations.append(op_tuple)

                        transformer = MathematicalOperations([op_tuple])
                        x_transformed = transformer.fit_transform(x)

                        new_col = next(
                            c for c in x_transformed.columns if c not in x.columns
                        )
                        transformations[new_col] = op_tuple

        return transformations, operations

    def _create_blocks(self, operations, block_size):
        return [
            operations[i : i + block_size]
            for i in range(0, len(operations), block_size)
        ]

    def _evaluate_blocks(self, x, y, model, blocks, transformations_map, logger):
        selected = []

        for i, block in enumerate(blocks, start=1):
            logger.task_update(f"Evaluating block {i}/{len(blocks)}")
            selected_block = []

            transformer = MathematicalOperations(block)
            x_transformed = transformer.fit_transform(x)
            selected_cols, scores = ProbeFeatureSelector.fit(x_transformed, y, model)

            for col in selected_cols:
                if col not in transformations_map:
                    continue

                col1, col2, _ = transformations_map[col]
                if scores.get(col, 0) > max(scores.get(col1, 0), scores.get(col2, 0)):
                    selected_block.append(transformations_map[col])

            selected.extend(selected_block)
            logger.progress(f"   ↪ Block {i}: {len(selected_block)} selected features")

        return selected

    def _refine_operations(
        self,
        x,
        y,
        model,
        scoring,
        direction,
        cv,
        groups,
        operations,
        transformations_map,
    ):
        transformer = MathematicalOperations(operations)
        x_transformed = transformer.fit_transform(x, y)

        rfa = RecursiveFeatureAddition(model, scoring, direction, cv, groups)
        final_selected_cols = rfa.fit(x_transformed, y)

        return [
            transformations_map[col]
            for col in final_selected_cols
            if col in transformations_map
        ]
