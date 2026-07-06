from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class ScaleTransformation(BaseEstimator, TransformerMixin):
    def __init__(
        self, transformation_options=None, quantile_range=None, track_columns=False
    ):
        self.transformation_options = transformation_options
        self.quantile_range = quantile_range
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._transformers = {}

    def fit(self, X, y=None):
        self._transformers = {}

        for column, transformation in (self.transformation_options or {}).items():
            if transformation == "min_max":
                transformer = MinMaxScaler()

            elif transformation == "standard":
                transformer = StandardScaler()

            elif transformation == "robust":
                quantile_range = self.quantile_range.get(column, (25.0, 75.0))
                transformer = RobustScaler(quantile_range=quantile_range)

            elif transformation == "max_abs":
                transformer = MaxAbsScaler()

            else:
                continue

            self._transformers[column] = transformer.fit(X[[column]])

        return self

    def transform(self, X, y=None):
        X = X.copy()

        for column, scaler in self._transformers.items():
            X[[column]] = scaler.transform(X[[column]])

            if self.track_columns:
                self.tracked_columns[column] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
