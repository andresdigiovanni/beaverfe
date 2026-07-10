from itertools import product

import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class NumericalBinningSpaceGenerator(PerColumnSpaceGenerator):
    prefix = "binning"
    STRATEGIES = ["quantile", "uniform"]
    BIN_COUNTS = [5, 10]

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        n_unique = X[col].nunique()
        return [
            f"{strategy}__{n_bins}"
            for strategy, n_bins in product(self.STRATEGIES, self.BIN_COUNTS)
            if n_unique > n_bins
        ]
