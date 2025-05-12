import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MathematicalOperations(BaseEstimator, TransformerMixin):
    def __init__(self, operations_options=None, track_columns=False):
        self.operations_options = operations_options or []
        self.track_columns = track_columns

        self.tracked_columns = {}

    def get_params(self, deep=True):
        return {
            "operations_options": self.operations_options,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        # No fitting required for this transformer, maintaining compatibility with scikit-learn API
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for col1, col2, operation in self.operations_options:
            new_column = f"{col1}__{operation}__{col2}"

            if operation == "add":
                X[new_column] = X[col1] + X[col2]

            elif operation == "subtract":
                X[new_column] = X[col1] - X[col2]

            elif operation == "multiply":
                X[new_column] = X[col1] * X[col2]

            elif operation == "divide":
                X[new_column] = X[col1] / X[col2]

            elif operation == "modulus":
                X[new_column] = X[col1] % X[col2]

            elif operation == "hypotenuse":
                X[new_column] = np.hypot(X[col1], X[col2])

            elif operation == "mean":
                X[new_column] = (X[col1] + X[col2]) / 2

            # Prevent NaNs
            X[new_column] = X[new_column].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Prevent fragmentation by explicitly copying the DataFrame
            X = X.copy()

            if self.track_columns:
                self.tracked_columns[new_column] = [col1, col2]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
