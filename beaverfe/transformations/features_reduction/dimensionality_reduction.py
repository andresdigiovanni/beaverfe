import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding


class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(
        self, features=None, method=None, n_components=None, track_columns=False
    ):
        self.features = features
        self.method = method
        self.n_components = n_components
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._reducer = None

    def get_params(self, deep=True):
        return {
            "features": self.features,
            "method": self.method,
            "n_components": self.n_components,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        X = X.copy()
        X = X[self.features]

        self._reducer = None

        if self.method == "factor_analysis":
            self._reducer = FactorAnalysis(n_components=self.n_components).fit(X)

        elif self.method == "ica":
            self._reducer = FastICA(n_components=self.n_components).fit(X)

        elif self.method == "isomap":
            self._reducer = Isomap(n_components=self.n_components).fit(X)

        elif self.method == "kernel_pca":
            self._reducer = KernelPCA(n_components=self.n_components, kernel="rbf").fit(
                X
            )

        elif self.method == "lda":
            if y is None:
                raise ValueError("LDA requires target values (y)")
            self._reducer = LinearDiscriminantAnalysis(
                n_components=self.n_components
            ).fit(X, y)

        elif self.method == "lle":
            self._reducer = LocallyLinearEmbedding(
                n_components=self.n_components, eigen_solver="dense"
            ).fit(X)

        elif self.method == "pca":
            self._reducer = PCA(n_components=self.n_components).fit(X)

        elif self.method == "truncated_svd":
            self._reducer = TruncatedSVD(n_components=self.n_components).fit(X)

        else:
            raise ValueError(f"Unknown reduction method: {self.method}")

        return self

    def transform(self, X, y=None):
        X_subset = X[self.features]

        reduced_array = self._reducer.transform(X_subset)
        columns = [f"{self.method}_{i + 1}" for i in range(reduced_array.shape[1])]

        reduced_df = pd.DataFrame(
            reduced_array,
            columns=columns,
            index=X.index,
        )

        if self.track_columns:
            for column in columns:
                self.tracked_columns[column] = list(self.features)

        X = pd.concat([X.drop(columns=self.features), reduced_df], axis=1)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
