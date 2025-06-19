from typing import Any, Dict, Optional, Tuple

from beaverfe.auto_parameters.shared import evaluate_model
from beaverfe.auto_parameters.shared.utils import is_score_improved
from beaverfe.transformations import DimensionalityReduction
from beaverfe.utils.verbose import VerboseLogger


class DimensionalityReductionParameterSelector:
    def select_best_parameters(
        self,
        X,
        y,
        model,
        scoring,
        direction: str,
        cv,
        groups,
        tol,
        logger: VerboseLogger,
    ) -> Optional[Dict[str, Any]]:
        logger.task_start("Starting dimensionality reduction")

        n_features = X.shape[1]
        n_classes = y.nunique()

        if n_features < 2:
            logger.warn("No dimensionality reduction was applied: less than 2 columns")
            return None

        best_method = None
        best_n_components = None
        best_score = evaluate_model(X, y, model, scoring, cv, groups)
        logger.baseline(f"Base score: {best_score:.4f}")

        methods = self._get_applicable_methods(X)

        for method in methods:
            max_components = min(50, n_features)
            if method == "lda":
                max_components = min(max_components, n_classes - 1)

            n_components, score = self._search_optimal_components(
                X, y, method, (2, max_components), model, scoring, direction, cv, groups
            )
            logger.progress(f"   ↪ Tried '{method}' → Score: {score:.4f}")

            if is_score_improved(score, best_score, direction, tol):
                best_score = score
                best_method = method
                best_n_components = n_components

        if best_method:
            transformer = DimensionalityReduction(
                features=list(X.columns),
                method=best_method,
                n_components=best_n_components,
            )
            logger.task_result(
                f"Best method: {best_method} with {best_n_components} components"
            )
            return {
                "name": transformer.__class__.__name__,
                "params": transformer.get_params(),
            }

        logger.warn("No dimensionality reduction was applied")
        return None

    def _get_applicable_methods(self, X):
        return ["lda", "pca", "truncated_svd"]

    def _search_optimal_components(
        self,
        X,
        y,
        method: str,
        n_range: Tuple[int, int],
        model,
        scoring,
        direction: str,
        cv,
        groups,
    ) -> Tuple[int, float]:
        low, high = n_range
        best_n = low
        best_score = float("-inf") if direction == "maximize" else float("inf")
        scores = {}

        while low <= high:
            mid1 = low + (high - low) // 3
            mid2 = high - (high - low) // 3

            for mid in [mid1, mid2]:
                if mid not in scores:
                    transformer = DimensionalityReduction(
                        features=list(X.columns),
                        method=method,
                        n_components=mid,
                    )
                    scores[mid] = evaluate_model(
                        X, y, model, scoring, cv, groups, transformer
                    )

            score1, score2 = scores[mid1], scores[mid2]

            if is_score_improved(score1, best_score, direction):
                best_score = score1
                best_n = mid1

            if is_score_improved(score2, best_score, direction):
                best_score = score2
                best_n = mid2

            if is_score_improved(score2, score1, direction):
                low = mid1 + 1
            else:
                high = mid2 - 1

        return best_n, best_score
