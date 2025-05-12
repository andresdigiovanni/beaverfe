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

    def select_best_parameters(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        numerical_columns = dtypes.numerical_columns(x)
        column_indices = range(len(numerical_columns))
        total_columns = len(numerical_columns)

        all_transformations = []
        all_selected_features = []

        logger.task_start("Starting mathematical operations search")

        for i, idx1 in enumerate(column_indices, start=1):
            col1 = numerical_columns[idx1]
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{col1}'")

            transformations, operation_options = self._generate_operations(
                x, col1, idx1, numerical_columns, column_indices
            )
            all_transformations.extend(transformations)

            transformer = MathematicalOperations(operation_options)
            x_transformed = transformer.fit_transform(x, y)

            selected = ProbeFeatureSelector.fit(x_transformed, y, model)
            all_selected_features.extend(selected)
            n_selected = len(set(selected) - set(x.columns))

            logger.progress(
                f"   ↪ Evaluated {len(transformations)} operations → Selected: {n_selected}"
            )

        selected_operations = self._filter_selected_transformations(
            all_transformations, all_selected_features
        )

        if selected_operations:
            logger.task_update(
                "Refining selected operations using Recursive Feature Addition"
            )
            x_transformed = MathematicalOperations(selected_operations).fit_transform(
                x, y
            )

            rfa = RecursiveFeatureAddition(model, scoring, direction, cv, groups)
            refined_features = rfa.fit(x_transformed, y)

            selected_operations = self._filter_selected_transformations(
                all_transformations, refined_features
            )

        if selected_operations:
            logger.task_result(
                f"Selected {len(selected_operations)} mathematical operation(s)"
            )
            transformer = MathematicalOperations(selected_operations)
            return {
                "name": transformer.__class__.__name__,
                "params": transformer.get_params(),
            }

        logger.warn("No mathematical operations were applied")
        return None

    def _generate_operations(self, x, col1, idx1, columns, indices):
        transformations = []
        options = []

        for op in self.SYMMETRIC_OPERATIONS:
            t, o = self._create_operations(
                x, col1, idx1, columns, indices, op, symmetric=True
            )
            transformations.extend(t)
            options.extend(o)

        for op in self.NON_SYMMETRIC_OPERATIONS:
            t, o = self._create_operations(
                x, col1, idx1, columns, indices, op, symmetric=False
            )
            transformations.extend(t)
            options.extend(o)

        return transformations, options

    def _create_operations(self, x, col1, idx1, columns, indices, operation, symmetric):
        transformations = []
        options = []

        for idx2 in indices:
            if (symmetric and idx1 >= idx2) or (not symmetric and idx1 == idx2):
                continue

            col2 = columns[idx2]
            operation_option = (col1, col2, operation)
            options.append(operation_option)

            transformer = MathematicalOperations([operation_option])
            x_transformed = transformer.fit_transform(x)

            # Assume new column is the one not in original set
            new_column_name = next(
                col for col in x_transformed.columns if col not in x.columns
            )

            transformations.append(
                {
                    "operation_option": operation_option,
                    "transformed_column": new_column_name,
                }
            )

        return transformations, options

    def _filter_selected_transformations(self, transformations_info, selected_features):
        return [
            info["operation_option"]
            for info in transformations_info
            if info["transformed_column"] in selected_features
        ]
