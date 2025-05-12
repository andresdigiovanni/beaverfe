import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutliersHandler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        transformation_options=None,
        thresholds=None,
        lof_params=None,
        iforest_params=None,
        track_columns=False,
    ):
        self.transformation_options = transformation_options
        self.thresholds = thresholds
        self.lof_params = lof_params
        self.iforest_params = iforest_params
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._statistics = {}
        self._bounds = {}
        self._handlers = {}

    def get_params(self, deep=True):
        return {
            "transformation_options": self.transformation_options,
            "thresholds": self.thresholds,
            "lof_params": self.lof_params,
            "iforest_params": self.iforest_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._statistics = {}
        self._bounds = {}
        self._handlers = {}

        for column, (action, method) in self.transformation_options.items():
            # Specific methods fit
            if method == "lof":
                self._apply_lof(X, column)

            elif method == "iforest":
                self._apply_iforest(X, column)

            elif method in ["iqr", "zscore"]:
                lower_bound, upper_bound = self._calculate_bounds(X, column, method)
                self._bounds[column] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }

            # Specific actions fit
            if action == "median":
                self._statistics[column] = X[column].median()

        return self

    def _apply_lof(self, df, column):
        lof = LocalOutlierFactor(**self.lof_params[column], novelty=True)
        lof.fit(np.array(df[[column]]))
        self._handlers[column] = lof

    def _apply_iforest(self, df, column):
        iforest = IsolationForest(**self.iforest_params[column])
        iforest.fit(np.array(df[[column]]))
        self._handlers[column] = iforest

    def _calculate_bounds(self, df, column, method):
        if method == "iqr":
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - self.thresholds[column] * iqr
            upper_bound = q3 + self.thresholds[column] * iqr

        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()

            lower_bound = mean - self.thresholds[column] * std
            upper_bound = mean + self.thresholds[column] * std

        else:
            lower_bound, upper_bound = 0, 0

        return lower_bound, upper_bound

    def transform(self, X, y=None):
        X = X.copy()

        for column, (action, method) in self.transformation_options.items():
            lower_bound, upper_bound = 0, 0

            if method in ["iqr", "zscore"]:
                lower_bound = self._bounds[column]["lower_bound"]
                upper_bound = self._bounds[column]["upper_bound"]

            if action == "cap":
                X[column] = np.clip(X[column], lower_bound, upper_bound)

            elif action == "median":
                if method in ["iforest", "lof"]:
                    handler = self._handlers[column]
                    y_pred = handler.predict(np.array(X[[column]].dropna()))
                    outliers = y_pred == -1

                elif method in ["iqr", "zscore"]:
                    outliers = (X[column] < lower_bound) | (X[column] > upper_bound)

                X[column] = np.where(
                    outliers,
                    self._statistics[column],
                    X[column],
                )

            if self.track_columns:
                self.tracked_columns[column] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
