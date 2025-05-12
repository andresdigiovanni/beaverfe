from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer

from beaverfe.transformations.utils import dtypes


class MissingValuesHandler(BaseEstimator, TransformerMixin):
    def __init__(
        self, transformation_options=None, n_neighbors=None, track_columns=False
    ):
        self.transformation_options = transformation_options
        self.n_neighbors = n_neighbors
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._statistics = {}
        self._imputers = {}

    def get_params(self, deep=True):
        return {
            "transformation_options": self.transformation_options,
            "n_neighbors": self.n_neighbors,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        self._statistics = {}
        self._imputers = {}

        for column, action in self.transformation_options.items():
            if action in ["mean", "median", "most_frequent"]:
                imputer = SimpleImputer(strategy=action)
                imputer.fit(X[[column]])
                self._imputers[column] = imputer

            elif action == "knn":
                imputer = KNNImputer(n_neighbors=self.n_neighbors.get(column, 5))
                imputer.fit(X[[column]])
                self._imputers[column] = imputer

        return self

    def transform(self, X, y=None):
        X = X.copy()
        cat_columns = dtypes.categorical_columns(X)

        for column, action in self.transformation_options.items():
            if action == "fill_0":
                fill_with = "Unknown" if column in cat_columns else 0
                X[column] = X[column].fillna(fill_with)

            elif action in ["mean", "median", "most_frequent", "knn"]:
                imputer = self._imputers[column]
                X[column] = imputer.transform(X[[column]]).flatten()

            if self.track_columns:
                self.tracked_columns[column] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
