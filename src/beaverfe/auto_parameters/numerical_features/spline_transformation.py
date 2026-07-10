from itertools import product

import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class SplineTransformationSpaceGenerator(PerColumnSpaceGenerator):
    prefix = "spline"
    N_KNOTS_OPTIONS = [5, 10]
    DEGREE_OPTIONS = [2, 3]
    EXTRAPOLATION_OPTIONS = ["linear"]

    def _options_for_column(self, X: pd.DataFrame, col: str) -> list[str]:
        return [
            f"{n_knots}__{degree}__{extrapolation}"
            for n_knots, degree, extrapolation in product(
                self.N_KNOTS_OPTIONS, self.DEGREE_OPTIONS, self.EXTRAPOLATION_OPTIONS
            )
        ]
