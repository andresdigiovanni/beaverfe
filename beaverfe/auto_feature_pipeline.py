from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np
from sklearn.utils import shuffle

import beaverfe.auto_parameters as pc
from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import ColumnSelection
from beaverfe.transformations.utils import dtypes
from beaverfe.utils import get_transformer
from beaverfe.utils.verbose import VerboseLogger


def auto_feature_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    model,
    scoring: str,
    direction: str = "maximize",
    cv: Union[int, Callable] = None,
    groups: Optional[np.ndarray] = None,
    tol: float = 0.005,
    verbose: bool = True,
    preprocessing: bool = True,
    feature_generation: bool = True,
    normalization: bool = True,
    dimensionality_reduction: bool = True,
) -> List[dict]:
    """
    Automatically applies a series of data transformations to improve model performance.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable.
        model: Machine learning model with a fit method.
        scoring (str): Scoring metric for evaluation.
        direction (str, optional): "maximize" or "minimize". Defaults to "maximize".
        cv (Union[int, Callable], optional): Cross-validation strategy. Defaults to None.
        groups (Optional[np.ndarray], optional): Group labels for cross-validation splitting. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        preprocessing (bool, optional): Whether to apply preprocessing steps (missing values, outliers, datetime). Defaults to True.
        feature_generation (bool, optional): Whether to apply feature generation (splines, binning, math ops, categorical encoding). Defaults to True.
        normalization (bool, optional): Whether to apply normalization or transformation of distributions. Defaults to True.
        dimensionality_reduction (bool, optional): Whether to apply column selection and dimensionality reduction. Defaults to True.

    Returns:
        List[dict]: A list of transformation steps that were applied, including their names and parameters.
    """

    logger = VerboseLogger(verbose)

    logger.header("Starting automated transformation search")
    logger.config(f"Input shape: {X.shape}")
    logger.config(f"Model: {model.__class__.__name__}")
    logger.config(f"Scoring metric: '{scoring}' with direction '{direction}'")

    X, y = X.copy(), y.copy()
    X, y = shuffle(X, y, random_state=42)

    initial_columns = X.columns.tolist()
    initial_num_columns = dtypes.numerical_columns(X)
    transformations, tracked_columns = [], []

    def apply_wrapper(
        transform_calc, X, y, transformations, tracked_columns, subset=None
    ):
        X_t, new_transforms, new_tracks = execute_transformation(
            transform_calc,
            X,
            y,
            model,
            scoring,
            direction,
            cv,
            groups,
            tol,
            logger,
            subset,
        )
        transformations.extend(new_transforms)
        tracked_columns.extend(new_tracks)
        new_columns = set(X_t.columns) - set(X.columns)

        return X_t, transformations, tracked_columns, new_columns

    # Apply Missing and Outlier handling
    if preprocessing:
        transformer = pc.MissingValuesIndicatorParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer, X, y, transformations, tracked_columns
        )

        transformer = pc.MissingValuesHandlerParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer, X, y, transformations, tracked_columns
        )

        transformer = pc.OutliersParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer,
            X,
            y,
            transformations,
            tracked_columns,
            subset=initial_num_columns,
        )

    math_columns = set()

    # Feature Engineering
    if feature_generation:
        transformer = pc.SplineTransformationParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer,
            X,
            y,
            transformations,
            tracked_columns,
            subset=initial_num_columns,
        )

        transformer = pc.NumericalBinningParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer,
            X,
            y,
            transformations,
            tracked_columns,
            subset=initial_num_columns,
        )

        transformer = pc.MathematicalOperationsParameterSelector()
        X, transformations, tracked_columns, math_columns = apply_wrapper(
            transformer,
            X,
            y,
            transformations,
            tracked_columns,
            subset=initial_num_columns,
        )

    if normalization:
        # Columns to normalize: original numerical + generated math columns
        normalization_columns = list(set(initial_num_columns) | math_columns)

        # Distribution Transformations (choose best)
        transformations_1, transformations_2 = [], []
        tracked_columns_1, tracked_columns_2 = [], []

        ## Option 1: NonLinear + Normalization + Scaler
        transformer = pc.NonLinearTransformationParameterSelector()
        X_1, transformations_1, tracked_columns_1, _ = apply_wrapper(
            transformer,
            X,
            y,
            transformations_1,
            tracked_columns_1,
            subset=normalization_columns,
        )

        transformer = pc.NormalizationParameterSelector()
        X_1, transformations_1, tracked_columns_1, _ = apply_wrapper(
            transformer,
            X_1,
            y,
            transformations_1,
            tracked_columns_1,
            subset=normalization_columns,
        )

        transformer = pc.ScaleTransformationParameterSelector()
        X_1, transformations_1, tracked_columns_1, _ = apply_wrapper(
            transformer,
            X_1,
            y,
            transformations_1,
            tracked_columns_1,
            subset=normalization_columns,
        )

        ## Option 2: Quantile Transformation
        transformer = pc.QuantileTransformationParameterSelector()
        X_2, transformations_2, tracked_columns_2, _ = apply_wrapper(
            transformer,
            X,
            y,
            transformations_2,
            tracked_columns_2,
            subset=normalization_columns,
        )

        ## Choose best transformation approach
        score_base = evaluate_model(X, y, model, scoring, cv, groups)
        score_1 = evaluate_model(X_1, y, model, scoring, cv, groups)
        score_2 = evaluate_model(X_2, y, model, scoring, cv, groups)

        if is_score_improved(score_1, score_base, direction, tol) or is_score_improved(
            score_2, score_base, direction, tol
        ):
            if is_score_improved(score_1, score_2, direction):
                X = X_1
                transformations.extend(transformations_1)
                tracked_columns.extend(tracked_columns_1)
                logger.task_result(
                    "Selected transformation strategy: Option 1 (NonLinear + Normalization + Scaler)"
                )
            else:
                X = X_2
                transformations.extend(transformations_2)
                tracked_columns.extend(tracked_columns_2)
                logger.task_result(
                    "Selected transformation strategy: Option 2 (QuantileTransformer)"
                )
        else:
            logger.warn("No transformation strategy improved performance.")

    if preprocessing:
        # Periodic Features
        transformer = pc.DateTimeTransformerParameterSelector()
        X, transformations, tracked_columns, datetime_columns = apply_wrapper(
            transformer, X, y, transformations, tracked_columns
        )
    else:
        datetime_columns = []

    if feature_generation:
        # Cyclical features
        if datetime_columns:
            transformer = pc.CyclicalFeaturesTransformerParameterSelector()
            X, transformations, tracked_columns, _ = apply_wrapper(
                transformer,
                X,
                y,
                transformations,
                tracked_columns,
                subset=list(datetime_columns),
            )

        # Categorical Encoding
        transformer = pc.CategoricalEncodingParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer, X, y, transformations, tracked_columns
        )

    # Dimensionality Reduction
    if dimensionality_reduction:
        transformer = pc.ColumnSelectionParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer,
            X,
            y,
            transformations,
            tracked_columns,
        )

        transformer = pc.DimensionalityReductionParameterSelector()
        X, transformations, tracked_columns, _ = apply_wrapper(
            transformer,
            X,
            y,
            transformations,
            tracked_columns,
        )

    # Remove unnecessary tranformations
    final_columns = set(X.columns)
    return filter_transformations(
        transformations, tracked_columns, initial_columns, final_columns
    )


def date_time() -> str:
    """Returns the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def execute_transformation(
    calculator,
    X,
    y,
    model,
    scoring,
    direction,
    cv,
    groups,
    tol,
    logger,
    subset=None,
):
    X_subset = X.loc[:, subset] if subset else X

    transformation = calculator.select_best_parameters(
        X_subset, y, model, scoring, direction, cv, groups, tol, logger
    )
    if not transformation:
        return X, [], []

    transformer = get_transformer(
        transformation["name"], {**transformation["params"], "track_columns": True}
    )
    X_transformed = transformer.fit_transform(X, y)
    return X_transformed, [transformation], [transformer.tracked_columns]


def filter_transformations(
    transformations, column_dependencies, initial_columns, target_columns
):
    # Initially, we only need the columns that appear in the final dataset
    required_columns = set(target_columns)
    filtered_transformations = []

    # Traverse transformations in reverse order to check backward dependencies
    for i in reversed(range(len(transformations))):
        transformation = transformations[i]
        name = transformation["name"]
        params = transformation["params"].copy()
        dependencies = column_dependencies[i]

        # Determine if any output column from this transformation is required
        relevant_outputs = [out for out in dependencies if out in required_columns]
        if not relevant_outputs:
            continue  # Skip this transformation if it doesn't contribute any required columns

        # Add source columns needed to generate the required output columns
        for out_col in relevant_outputs:
            required_columns.update(dependencies[out_col])

        # Filter out unnecessary parameters
        if "features" in params:
            params["features"] = [
                col for col in params["features"] if col in required_columns
            ]
            if not params["features"]:
                continue

        elif "transformation_options" in params:
            params["transformation_options"] = {
                k: v
                for k, v in params["transformation_options"].items()
                if k in required_columns
            }
            if not params["transformation_options"]:
                continue

        elif "operations_options" in params:
            params["operations_options"] = [
                (a, b, op)
                for a, b, op in params["operations_options"]
                if a in required_columns and b in required_columns
            ]
            if not params["operations_options"]:
                continue

        filtered_transformations.append({"name": name, "params": params})

    # Finally, add a ColumnSelection with the initial columns that are still required
    selected_initial_columns = [
        col for col in initial_columns if col in required_columns
    ]
    if selected_initial_columns:
        selector = ColumnSelection(selected_initial_columns)
        filtered_transformations.append(
            {
                "name": selector.__class__.__name__,
                "params": selector.get_params(),
            }
        )

    return filtered_transformations[::-1]
