import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score


class PermutationRFECV(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator,
        scoring=None,
        cv=None,
        min_features_to_select=1,
        step=1,
        n_repeats=5,
        random_state=None,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        # Store feature names
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
            X = X.values
        else:
            X = np.asarray(X)
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])

        n_features = X.shape[1]
        self.n_total_features_ = n_features
        self.support_ = np.ones(n_features, dtype=bool)
        self.ranking_ = np.ones(n_features, dtype=int)
        self.scores_ = []

        current_features = np.arange(n_features)

        while len(current_features) >= self.min_features_to_select:
            # Evaluate cross-validated performance
            estimator = clone(self.estimator)
            cv_score = np.mean(
                cross_val_score(
                    estimator,
                    X[:, current_features],
                    y,
                    scoring=self.scoring,
                    cv=self.cv,
                    groups=groups,
                )
            )
            self.scores_.append((current_features.copy(), cv_score))

            if len(current_features) == self.min_features_to_select:
                break

            # Fit on current features
            estimator.fit(X[:, current_features], y)

            # Compute permutation importance
            result = permutation_importance(
                estimator,
                X[:, current_features],
                y,
                scoring=self.scoring,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            importances = result.importances_mean
            ranking_indices = np.argsort(importances)

            # Determine how many features to remove
            if isinstance(self.step, float) and 0 < self.step < 1:
                n_remove = int(max(1, self.step * len(current_features)))
            else:
                n_remove = min(
                    int(self.step), len(current_features) - self.min_features_to_select
                )

            remove_indices = ranking_indices[:n_remove]

            self.ranking_[current_features[remove_indices]] = np.max(self.ranking_) + 1
            self.support_[current_features[remove_indices]] = False
            current_features = np.delete(current_features, remove_indices)

        # Select the best-performing subset
        best_features, best_score = max(self.scores_, key=lambda x: x[1])
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[best_features] = True
        self.n_features_ = self.support_.sum()
        self.grid_scores_ = [score for _, score in self.scores_]

        # Fit final model
        self.estimator_ = clone(self.estimator).fit(X[:, self.support_], y)
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.loc[:, self.support_]
        return X[:, self.support_]

    def fit_transform(self, X, y, groups=None):
        return self.fit(X, y, groups=groups).transform(X)

    def get_support(self, indices=False):
        return np.where(self.support_)[0] if indices else self.support_

    def get_feature_names_out(self):
        return self.feature_names_in_[self.support_]

    def score(self, X, y, groups=None):
        if hasattr(X, "iloc"):
            X_subset = X.loc[:, self.support_]
        else:
            X_subset = X[:, self.support_]
        return np.mean(
            cross_val_score(
                self.estimator_,
                X_subset,
                y,
                scoring=self.scoring,
                cv=self.cv,
                groups=groups,
            )
        )


if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_wine(return_X_y=True, as_frame=True)

    selector = PermutationRFECV(
        estimator=RandomForestClassifier(random_state=42),
        scoring="accuracy",
        step=0.2,  # 20% removal at each step
    )

    X_sel = selector.fit_transform(X, y)
    print("Selected features:", selector.get_feature_names_out())
    print("Final CV score:", selector.score(X, y))
