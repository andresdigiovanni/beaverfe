from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


class QuantileTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None, track_columns=False):
        self.transformation_options = transformation_options
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._transformers = {}
        self._n_quantiles = 1000

    def fit(self, X, y=None):
        self._transformers = {}

        n_samples = X.shape[0]
        self._n_quantiles = min(1000, n_samples)

        for column, transformation in (self.transformation_options or {}).items():
            if transformation in ["uniform", "normal"]:
                transformer = QuantileTransformer(
                    n_quantiles=self._n_quantiles, output_distribution=transformation
                )
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
