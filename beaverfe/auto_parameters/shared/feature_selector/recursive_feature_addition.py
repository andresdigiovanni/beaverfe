from typing import Callable, Optional, Union

import numpy as np
from sklearn.model_selection import cross_val_score

from beaverfe.auto_parameters.shared.feature_selector.shared import feature_importance
from beaverfe.utils.verbose import VerboseLogger


class RecursiveFeatureAddition:
    def __init__(
        self,
        model,
        scoring: str,
        direction: str = "maximize",
        cv: Union[int, Callable] = None,
        groups: Optional = None,
        early_stopping: int = 3,
        target_score: bool = True,
        logger: VerboseLogger = None,
    ):
        """
        Initializes the RecursiveFeatureAddition class.

        Args:
            model: Machine learning model with a fit method.
            scoring (str): Scoring metric for evaluation.
            direction (str, optional): "maximize" to increase score or "minimize" to decrease. Defaults to "maximize".
            cv (Union[int, Callable], optional): Number of beaverfe-validation folds or a custom beaverfe-validation generator. Defaults to None.
            groups (Optional): Group labels for beaverfe-validation splitting. Defaults to None.
            early_stopping (int, optional): Maximum number of non-improving additions. Defaults to 3.
            target_score (bool, optional): Whether to use the full-feature score as a stopping criterion. Defaults to True.
            logger (VerboseLogger): Logger for structured output.
        """
        self.model = model
        self.scoring = scoring
        self.direction = direction
        self.cv = cv
        self.groups = groups
        self.early_stopping = early_stopping
        self.target_score = target_score
        self.logger = logger

        self.scores_history = []
        self.target_score_value = None
        self.selected_features_names = []

    def _log(self, message: str):
        if self.logger:
            self.logger.progress(message)

    def fit(self, X: np.ndarray, y: np.ndarray) -> list:
        X = X.copy()

        scores = cross_val_score(
            self.model,
            X,
            y,
            scoring=self.scoring,
            cv=self.cv,
            groups=self.groups,
            n_jobs=-1,
        )
        self.target_score_value = np.mean(scores)
        if self.logger:
            self.logger.baseline(
                f"Target score (all features): {self.target_score_value:.4f}"
            )

        self.model.fit(X, y)
        feature_importances = feature_importance(self.model, X, y)
        feature_indices = np.argsort(feature_importances)[::-1]

        selected_features_idx = self._evaluate_features(X, y, feature_indices)
        self.selected_features_names = [X.columns[i] for i in selected_features_idx]

        return self.selected_features_names

    def _evaluate_features(
        self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray
    ) -> list:
        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        selected_features_idx = []
        features_added_without_improvement = 0

        for i, idx in enumerate(feature_indices):
            current_features_idx = selected_features_idx + [idx]

            scores = cross_val_score(
                self.model,
                X.iloc[:, current_features_idx],
                y,
                scoring=self.scoring,
                cv=self.cv,
                groups=self.groups,
                n_jobs=-1,
            )
            score = np.mean(scores)

            if self._is_score_improved(score, best_score, self.direction):
                self.scores_history.append(score)
                selected_features_idx.append(idx)
                best_score = score
                features_added_without_improvement = 0
                self._log(
                    f"   {i + 1}/{len(feature_indices)} ↪ Added '{X.columns[idx]}' → Score: {score:.4f}"
                )
            else:
                features_added_without_improvement += 1
                self._log(
                    f"   {i + 1}/{len(feature_indices)} ↪ Skipped '{X.columns[idx]}' → Score: {score:.4f}"
                )

            if self.target_score and not self._is_score_improved(
                score, self.target_score_value, self.direction
            ):
                continue

            if features_added_without_improvement >= self.early_stopping:
                self._log("Early stopping triggered.")
                break

        return selected_features_idx

    def _is_score_improved(
        self, score: float, best_score: float, direction: str
    ) -> bool:
        return (direction == "maximize" and score > best_score) or (
            direction == "minimize" and score < best_score
        )


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = KNeighborsClassifier()
    rfa = RecursiveFeatureAddition(model, scoring="roc_auc_ovr", verbose=True)
    selected_features = rfa.fit(X, y)
    print("Selected Features:", selected_features)
