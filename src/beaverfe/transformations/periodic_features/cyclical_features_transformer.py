import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None, track_columns=False):
        self.transformation_options = transformation_options
        self.track_columns = track_columns

        self.tracked_columns = {}

    def fit(self, X, y=None):
        return self  # No fitting necessary, but required for compatibility

    def transform(self, X, y=None):
        X = X.copy()

        for column, period in (self.transformation_options or {}).items():
            X[f"{column}_sin"] = np.sin(2 * np.pi * X[column] / period)
            X[f"{column}_cos"] = np.cos(2 * np.pi * X[column] / period)

            if self.track_columns:
                self.tracked_columns[f"{column}_sin"] = [column]
                self.tracked_columns[f"{column}_cos"] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
