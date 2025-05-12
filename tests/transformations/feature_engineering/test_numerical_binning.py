
import pandas as pd
import pytest

from beaverfe.transformations import NumericalBinning


class TestNumericalBinning:

    # Initializes with default binning options when none are provided
    def test_initialization_with_default_binning_options(self):
        binning = NumericalBinning()

        assert binning.binning_options == []

    # Handles empty binning_options without errors
    def test_handling_empty_binning_options(self):
        data = pd.DataFrame({'feature1': [1, 2, 3]})
        binning = NumericalBinning(binning_options=[])
        transformed_data = binning.fit_transform(data)
        
        pd.testing.assert_frame_equal(transformed_data, data)

    # Correctly sets and retrieves parameters using set_params and get_params
    def test_set_and_get_params(self):
        binning_options = [("column1", "uniform", 5), ("column2", "quantile", 10)]
        binning = NumericalBinning(binning_options=binning_options)

        params = binning.get_params()
        binning.set_params(binning_options=[("column3", "kmeans", 8)])

        updated_params = binning.get_params()

        assert params == {"binning_options": binning_options}
        assert updated_params == {"binning_options": [("column3", "kmeans", 8)]}

    # Successfully fits data using specified binning strategies and number of bins
    def test_fit_with_binning_strategies(self):
        binning_options = [("column1", "uniform", 5), ("column2", "quantile", 10)]
        X = pd.DataFrame({
            "column1": [1, 2, 3, None],
            "column2": [4, 5, None, 7]
        })

        binning = NumericalBinning(binning_options=binning_options)
        binning.fit(X)

        assert len(binning._binners) == 2
        assert "column1__uniform_5" in binning._binners
        assert "column2__quantile_10" in binning._binners