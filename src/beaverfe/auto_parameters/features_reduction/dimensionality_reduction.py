import numpy as np
import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator
from beaverfe.model_profiler import DataProfiler


class DimensionalityReductionSpaceGenerator(SpaceGenerator):
    N_COMPONENTS_OPTIONS = [2, 5, 10, 25, 50]

    def __init__(self) -> None:
        self._profiler: DataProfiler | None = None

    def set_profiler(self, profiler: DataProfiler) -> None:
        """Inject a :class:`~beaverfe.model_profiler.DataProfiler` to enable
        data-aware search space capping and method filtering.

        Args:
            profiler: Frozen profile computed from the training data.
        """
        self._profiler = profiler

    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        if X.shape[1] < 2:
            return {}

        y_series = y if isinstance(y, pd.Series) else pd.Series(y)
        n_classes = int(y_series.nunique())

        if self._profiler is not None:
            max_components = min(50, X.shape[1], self._profiler.n_samples // 3)
        else:
            max_components = min(50, X.shape[1])

        methods = self._get_applicable_methods(X)

        if not methods:
            return {}

        # Determine n_components options: filter to <= max_components
        component_options: list[int] = [
            n for n in self.N_COMPONENTS_OPTIONS if n <= max_components
        ]
        # For LDA, also filter to <= n_classes - 1
        has_lda = "lda" in methods
        if has_lda and not [n for n in component_options if n <= n_classes - 1]:
            # If no component fits LDA, remove lda
            methods = [m for m in methods if m != "lda"]

        if not methods:
            return {}

        if not component_options:
            return {}

        return {
            "dim_reduction_method": ["none", *methods],
            "dim_reduction_n_components": component_options,
        }

    def _get_applicable_methods(self, X: pd.DataFrame) -> list[str]:
        return ["lda", "pca", "truncated_svd"]
