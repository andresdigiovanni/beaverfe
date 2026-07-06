from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer


class Normalization(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None, track_columns=False):
        self.transformation_options = transformation_options
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._transformers = {}

    def fit(self, X, y=None):
        self._transformers = {}

        for column, transformation in (self.transformation_options or {}).items():
            if transformation in ["l1", "l2"]:
                transformer = Normalizer(norm=transformation)
                transformer.fit(X[[column]])
                self._transformers[column] = transformer

        return self

    def transform(self, X, y=None):
        X = X.copy()

        for column, transformer in self._transformers.items():
            X[column] = transformer.transform(X[[column]])

            if self.track_columns:
                self.tracked_columns[column] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
