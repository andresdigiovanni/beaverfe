from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelection(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, track_columns=False):
        self.features = features
        self.track_columns = track_columns

        self.tracked_columns = {}

    def get_params(self, deep=True):
        return {"features": self.features}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        # No fitting required, maintaining compatibility with scikit-learn API
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Select only the specified columns from X
        X = X[self.features]

        if self.track_columns:
            self.tracked_columns = {column: [column] for column in self.features}

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
