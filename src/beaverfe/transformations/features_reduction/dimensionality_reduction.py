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

    def fit(self, X, y=None):
        X = X.copy()
        X = X[self.features]

        self._reducer = None

        # Clamp n_components to the number of available features so that callers
        # never need to guard against the sklearn "n_components > n_features" error
        # (can happen when the feature list is pruned after the search space is built).
        n_components = min(self.n_components, X.shape[1])

        if self.method == "factor_analysis":
            self._reducer = FactorAnalysis(n_components=n_components).fit(X)

        elif self.method == "ica":
            self._reducer = FastICA(n_components=n_components).fit(X)

        elif self.method == "isomap":
            self._reducer = Isomap(n_components=n_components).fit(X)

        elif self.method == "kernel_pca":
            self._reducer = KernelPCA(n_components=n_components, kernel="rbf").fit(X)

        elif self.method == "lda":
            if y is None:
                raise ValueError("LDA requires target values (y)")
            self._reducer = LinearDiscriminantAnalysis(n_components=n_components).fit(
                X, y
            )

        elif self.method == "lle":
            # eigen_solver left at its sklearn default ("auto") rather than
            # forced to "dense": "dense" is an O(n^3) solve regardless of
            # dataset size, while "auto" picks a sparse/iterative solver for
            # larger inputs.
            self._reducer = LocallyLinearEmbedding(n_components=n_components).fit(X)

        elif self.method == "pca":
            self._reducer = PCA(n_components=n_components).fit(X)

        elif self.method == "truncated_svd":
            self._reducer = TruncatedSVD(n_components=n_components).fit(X)

        else:
            raise ValueError(f"Unknown reduction method: {self.method}")

        return self

    def transform(self, X, y=None):
        X_subset = X[self.features]

        assert self._reducer is not None
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
