from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import OutliersHandler
from beaverfe.transformations.utils import dtypes
from beaverfe.utils.verbose import VerboseLogger


class OutliersParameterSelector:
    def select_best_parameters(
        self, X, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        columns = dtypes.numerical_columns(X)
        total_columns = len(columns)
        outlier_methods = self._get_outlier_methods()
        outlier_actions = ["cap", "median"]

        best_params = {
            "transformation_options": {},
            "thresholds": {},
            "lof_params": {},
            "iforest_params": {},
        }

        logger.task_start("Starting outlier handling search")
        base_score = evaluate_model(X, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {base_score:.4f}")

        for i, column in enumerate(columns, start=1):
            logger.task_update(f"[{i}/{total_columns}] Evaluating column: '{column}'")

            best_column_params = self._find_best_params_for_column(
                X,
                y,
                model,
                scoring,
                direction,
                cv,
                groups,
                column,
                base_score,
                outlier_actions,
                outlier_methods,
                logger,
            )

            if best_column_params:
                logger.task_result(
                    f"Selected outlier handler for '{column}': {self._kwargs_to_string(best_column_params, column)}"
                )
                self._update_best_params(column, best_column_params, best_params)

        transformation_options = best_params["transformation_options"]
        if transformation_options:
            logger.task_result(
                f"Outlier handler applied to {len(transformation_options)} column(s)"
            )
            return self._build_outliers_handler(best_params)

        logger.warn("No outlier handler was applied to any column")
        return None

    def _get_outlier_methods(self):
        return {
            "iqr": {"thresholds": [1.5, 3.0]},
            "zscore": {"thresholds": [2.5, 3.0]},
            "iforest": {"contamination": [0.05, 0.1]},
        }

    def _find_best_params_for_column(
        self,
        X,
        y,
        model,
        scoring,
        direction,
        cv,
        groups,
        column,
        base_score,
        actions,
        methods,
        logger,
    ):
        best_score = base_score
        best_params = {}

        for action, method, param in self._generate_combinations(actions, methods):
            if not self._has_outliers(X[column], method, param):
                continue

            kwargs = self._build_kwargs(column, action, method, param)
            score = evaluate_model(
                X, y, model, scoring, cv, groups, OutliersHandler(**kwargs)
            )
            logger.progress(
                f"   ↪ Tried '{self._kwargs_to_string(kwargs, column)}' → Score: {score:.4f}"
            )

            if is_score_improved(score, best_score, direction):
                best_score = score
                best_params = kwargs

        return best_params

    def _generate_combinations(self, actions, methods):
        for method, params in methods.items():
            values = (
                params.get("n_neighbors")
                or params.get("contamination")
                or params.get("thresholds")
            )
            valid_actions = ["median"] if method == "iforest" else actions
            yield from product(valid_actions, [method], values)

    def _has_outliers(self, column_data, method, param):
        if method in ["iqr", "zscore"]:
            return self._get_outliers_count(column_data, method, param) > 0
        return self._get_outliers_count_ml(column_data, method, param) > 0

    def _get_outliers_count(self, data, method, threshold):
        clean = data.dropna()
        if method == "iqr":
            q1, q3 = np.percentile(clean, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        else:
            mean, std = clean.mean(), clean.std()
            lower, upper = mean - threshold * std, mean + threshold * std
        return clean[(clean < lower) | (clean > upper)].count()

    def _get_outliers_count_ml(self, data, method, param):
        clean = data.dropna().values.reshape(-1, 1)
        model = self._get_ml_model(method, param)
        preds = model.fit_predict(clean)
        return (preds == -1).sum()

    def _get_ml_model(self, method, param):
        if method == "lof":
            return LocalOutlierFactor(n_neighbors=param)
        return IsolationForest(contamination=param, random_state=42)

    def _build_kwargs(self, column, action, method, param):
        kwargs = {"transformation_options": {column: (action, method)}}
        if method == "lof":
            kwargs["lof_params"] = {column: {"n_neighbors": param}}
        elif method == "iforest":
            kwargs["iforest_params"] = {column: {"contamination": param}}
        else:
            kwargs["thresholds"] = {column: param}
        return kwargs

    def _kwargs_to_string(self, kwargs, column):
        action, method = kwargs["transformation_options"][column]
        if method == "lof":
            param = kwargs["lof_params"][column]["n_neighbors"]
            detail = f"n_neighbors: {param}"
        elif method == "iforest":
            param = kwargs["iforest_params"][column]["contamination"]
            detail = f"contamination: {param}"
        else:
            param = kwargs["thresholds"][column]
            detail = f"threshold: {param}"
        return f"action: {action}, method: {method}, {detail}"

    def _update_best_params(self, column, best_column_params, best_params):
        action = best_column_params["transformation_options"][column][0]
        if action != "none":
            for key in best_params:
                best_params[key].update(best_column_params.get(key, {}))

    def _build_outliers_handler(self, params):
        handler = OutliersHandler(
            transformation_options=params["transformation_options"],
            thresholds=params["thresholds"],
            lof_params=params["lof_params"],
            iforest_params=params["iforest_params"],
        )
        return {
            "name": handler.__class__.__name__,
            "params": handler.get_params(),
        }
