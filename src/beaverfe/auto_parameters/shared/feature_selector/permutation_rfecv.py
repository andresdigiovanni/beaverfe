import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split


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
        max_time_seconds=None,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.max_time_seconds = max_time_seconds

    def fit(self, X, y, groups=None):
        # Keep X as a DataFrame throughout so estimators that look up columns
        # by name (e.g. a nested feature-engineering pipeline) keep working
        # when we slice down to a subset of features.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            X = pd.DataFrame(np.asarray(X))
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])
            X.columns = self.feature_names_in_

        n_features = X.shape[1]
        self.n_total_features_ = n_features
        self.support_ = np.ones(n_features, dtype=bool)
        self.ranking_ = np.ones(n_features, dtype=int)
        self.scores_: list[tuple[np.ndarray, float]] = []

        # Hold out a validation slice used exclusively for permutation importance.
        # Using a separate split avoids overestimating importance on training data
        # (where a fitted model may have memorised the samples). The main
        # cross-validated score stored in self.scores_ still uses the full X/y,
        # so the final subset selection remains unbiased.
        X_train_imp, X_val_imp, y_train_imp, y_val_imp = train_test_split(
            X, np.asarray(y), test_size=0.2, random_state=self.random_state
        )

        current_features = np.arange(n_features)
        start_time = time.time()

        while len(current_features) >= self.min_features_to_select:
            # Stop elimination once the time budget is spent, keeping the
            # best subset found among rounds already completed. Only honored
            # once at least one round has scored something, so this never
            # leaves self.scores_ empty for the selection step below.
            if (
                self.max_time_seconds is not None
                and self.scores_
                and time.time() - start_time >= self.max_time_seconds
            ):
                break

            # Evaluate cross-validated performance
            estimator = clone(self.estimator)
            cv_score = np.mean(
                cross_val_score(
                    estimator,
                    X.iloc[:, current_features],
                    y,
                    scoring=self.scoring,
                    cv=self.cv,
                    groups=groups,
                )
            )
            self.scores_.append((current_features.copy(), cv_score))

            if len(current_features) == self.min_features_to_select:
                break

            # Fit on the importance-training split, then evaluate permutation
            # importance on the held-out validation split. This prevents the
            # estimator from inflating importance scores for features it simply
            # memorised on the training set.
            estimator = clone(self.estimator)
            estimator.fit(X_train_imp.iloc[:, current_features], y_train_imp)
            result = permutation_importance(
                estimator,
                X_val_imp.iloc[:, current_features],
                y_val_imp,
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

        # Select the best-performing subset.
        # When multiple rounds share the same (or nearly equal) top score,
        # prefer the subset with the fewest features — this avoids returning
        # all features in score-plateau situations (a common occurrence when the
        # dataset is small or the model is robust to redundant inputs).
        best_score = max(score for _, score in self.scores_)
        tol = max(abs(best_score) * 1e-4, 1e-6)
        best_features, _best_score = min(
            (
                (feats, score)
                for feats, score in self.scores_
                if best_score - score <= tol
            ),
            key=lambda x: len(x[0]),
        )
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[best_features] = True
        self.n_features_ = self.support_.sum()
        self.grid_scores_ = [score for _, score in self.scores_]

        # Fit final model
        self.estimator_ = clone(self.estimator).fit(X.iloc[:, self.support_], y)
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
