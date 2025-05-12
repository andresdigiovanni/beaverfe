import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class NonLinearTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None, track_columns=False):
        self.transformation_options = transformation_options
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._transformers = {}

    def get_params(self, deep=True):
        return {"transformation_options": self.transformation_options}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._transformers = {}

        for column, transformation in self.transformation_options.items():
            if transformation == "yeo_johnson":
                transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                transformer.fit(X[[column]])
                self._transformers[column] = transformer

        return self

    def transform(self, X, y=None):
        X = X.copy()

        for column, transformation in self.transformation_options.items():
            if transformation == "log":
                X[column] = np.log1p(X[column])

            elif transformation == "exponential":
                X[column] = np.exp(X[column])

            elif transformation == "yeo_johnson":
                transformer = self._transformers[column]
                X[column] = transformer.transform(X[[column]])

            if self.track_columns:
                self.tracked_columns[column] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
