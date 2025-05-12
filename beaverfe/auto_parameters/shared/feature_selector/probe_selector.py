import numpy as np
import pandas as pd
from sklearn.base import clone

from .shared import feature_importance


class ProbeFeatureSelector:
    @staticmethod
    def fit(X: pd.DataFrame, y, model, random_state: int = 42) -> list:
        """
        Selects features based on their importance compared to a random noise feature.

        Args:
            X (pd.DataFrame): Input features.
            y: Target variable.
            model: Machine learning model with a feature importance attribute or method.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            list: List of selected feature names.
        """
        X = X.copy()

        # Add a random noise feature
        np.random.seed(random_state)
        X["random_noise"] = np.random.random(size=len(X))

        converged = False

        while not converged:
            # Train a cloned model with the current dataset
            model_clone = clone(model)
            model_clone.fit(X, y)

            # Obtain feature importances
            importances = feature_importance(model_clone, X, y)

            feature_importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": importances}
            ).sort_values(by="importance", ascending=False)

            # Get the importance of the random noise feature
            noise_importance = feature_importance_df.loc[
                feature_importance_df["feature"] == "random_noise", "importance"
            ].iloc[0]

            # Identify features with importance greater than the random noise feature
            features_to_keep = feature_importance_df[
                feature_importance_df["importance"] > noise_importance
            ]["feature"].tolist()

            # Check if `random_noise` is the least important feature
            if feature_importance_df.iloc[-1]["feature"] == "random_noise":
                converged = True
            else:
                features_to_keep.append("random_noise")

            X = X[features_to_keep]

        return X.columns.tolist()
