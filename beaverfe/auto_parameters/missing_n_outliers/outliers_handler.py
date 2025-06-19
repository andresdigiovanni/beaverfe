from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import OutliersHandler
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class OutliersParameterSelector:
    def select_best_parameters(
        self, X, y, model, scoring, direction, cv, groups, tol, logger: VerboseLogger
    ):
        logger.task_start("Starting outlier handling search")

        columns = dtypes.numerical_columns(X)
        if not columns:
            logger.warn("No numerical columns found for outlier detection.")
            return None

        base_score = evaluate_model(X, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        best_params = {
            "transformation_options": {},
            "thresholds": {},
            "iforest_params": {},
        }

        outlier_methods = self._get_outlier_methods()
        actions = self._get_outlier_actions()

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{len(columns)}] Evaluating column: '{column}'")

            column_params = self._evaluate_column(
                X[column],
                X,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                tol,
                base_score,
                column,
                actions,
                outlier_methods,
                logger,
            )

            if column_params:
                self._update_best_params(column_params, best_params)
                logger.task_result(
                    f"Selected outlier handler: {self._kwargs_to_string(column_params, column)}"
                )

        transformation_options = best_params["transformation_options"]
        if transformation_options:
            logger.task_result(
                f"Outlier handler applied to {len(transformation_options)} column(s)"
            )
            return self._build_outliers_handler(best_params)

        logger.warn("No outlier handler was applied to any column.")
        return None

    def _get_outlier_methods(self):
        return {
            "iqr": {"thresholds": [1.5]},
            "zscore": {"thresholds": [3.0]},
            "iforest": {"contamination": [0.05]},
        }

    def _get_outlier_actions(self):
        return ["cap", "median"]

    def _evaluate_column(
        self,
        col_data,
        X,
        y,
        model,
        scoring,
        direction,
        cv,
        groups,
        tol,
        base_score,
        column,
        actions,
        methods,
        logger,
    ):
        best_score = base_score
        best_params = {}

        for action, method, param in self._generate_combinations(actions, methods):
            if not self._has_outliers(col_data, method, param):
                continue

            kwargs = self._build_kwargs(column, action, method, param)
            score = evaluate_model(
                X, y, model, scoring, cv, groups, OutliersHandler(**kwargs)
            )
            logger.progress(
                f"   ↪ Tried {self._kwargs_to_string(kwargs, column)} → Score: {score:.4f}"
            )

            if is_score_improved(score, best_score, direction, tol):
                best_score = score
                best_params = kwargs

        return best_params

    def _generate_combinations(self, actions, methods):
        for method, params in methods.items():
            values = next(iter(params.values()))
            valid_actions = ["median"] if method == "iforest" else actions

            for action, value in product(valid_actions, values):
                yield action, method, value

    def _has_outliers(self, data, method, param):
        clean = data.dropna()
        if method == "iqr":
            q1, q3 = np.percentile(clean, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - param * iqr, q3 + param * iqr

        elif method == "zscore":
            mean, std = clean.mean(), clean.std()
            lower, upper = mean - param * std, mean + param * std

        else:  # iforest
            preds = IsolationForest(contamination=param, random_state=42).fit_predict(
                clean.values.reshape(-1, 1)
            )
            return (preds == -1).sum() > 0

        return clean[(clean < lower) | (clean > upper)].count() > 0

    def _build_kwargs(self, column, action, method, param):
        kwargs = {"transformation_options": {column: (action, method)}}
        if method == "iforest":
            kwargs["iforest_params"] = {column: {"contamination": param}}
        else:
            kwargs["thresholds"] = {column: param}
        return kwargs

    def _kwargs_to_string(self, kwargs, column):
        action, method = kwargs["transformation_options"][column]

        if method == "iforest":
            param = kwargs["iforest_params"][column]["contamination"]
            param_str = f"contamination: {param}"

        else:
            param = kwargs["thresholds"][column]
            param_str = f"threshold: {param}"

        return f"action: {action}, method: {method}, {param_str}"

    def _update_best_params(self, new_params, best_params):
        for key in best_params:
            best_params[key].update(new_params.get(key, {}))

    def _build_outliers_handler(self, params):
        handler = OutliersHandler(
            transformation_options=params["transformation_options"],
            thresholds=params["thresholds"],
            iforest_params=params["iforest_params"],
        )
        return {
            "name": handler.__class__.__name__,
            "params": handler.get_params(),
        }
