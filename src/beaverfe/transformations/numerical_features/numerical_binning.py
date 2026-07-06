from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class NumericalBinning(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_options=None, track_columns=False):
        self.transformation_options = transformation_options
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._binners = {}

    def fit(self, X, y=None):
        self._binners = {}
        X = X.copy()

        for column, (strategy, n_bins) in (self.transformation_options or {}).items():
            X[column] = X[column].fillna(0)

            binner = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=strategy
            )
            binner.fit(X[[column]])
            self._binners[column] = binner

        return self

    def transform(self, X, y=None):
        X = X.copy()

        for column, (strategy, n_bins) in (self.transformation_options or {}).items():
            binner_name = f"{column}__bins_{strategy}_{n_bins}"
            binner = self._binners[column]

            binned_input = X[[column]].fillna(0)
            X[binner_name] = binner.transform(binned_input).flatten()

            if self.track_columns:
                self.tracked_columns[binner_name] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
