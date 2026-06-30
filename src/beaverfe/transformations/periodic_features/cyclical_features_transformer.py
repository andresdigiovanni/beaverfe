import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None, track_columns=False):
        self.transformation_options = transformation_options or {}
        self.track_columns = track_columns

        self.tracked_columns = {}

    def get_params(self, deep=True):
        return {
            "transformation_options": self.transformation_options,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        return self  # No fitting necessary, but required for compatibility

    def transform(self, X, y=None):
        X = X.copy()

        for column, period in self.transformation_options.items():
            X[f"{column}_sin"] = np.sin(2 * np.pi * X[column] / period)
            X[f"{column}_cos"] = np.cos(2 * np.pi * X[column] / period)

            if self.track_columns:
                self.tracked_columns[f"{column}_sin"] = [column]
                self.tracked_columns[f"{column}_cos"] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
