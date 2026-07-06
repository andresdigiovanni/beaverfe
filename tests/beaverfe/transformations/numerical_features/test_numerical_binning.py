import pandas as pd

from beaverfe.transformations import NumericalBinning


class TestNumericalBinning:
    def test_should_preserve_nan_in_untouched_columns_when_binning(self):
        # Arrange
        X = pd.DataFrame(
            {
                "binned_col": [1.0, 2.0, 3.0, 4.0],
                "other_col": [1.0, None, 3.0, None],
            }
        )
        binning = NumericalBinning(
            transformation_options={"binned_col": ("uniform", 2)}
        )

        # Act
        result = binning.fit_transform(X)

        # Assert
        assert result["other_col"].isna().sum() == 2

    # Initializes with default binning options when none are provided
    def test_initialization_with_default_binning_options(self):
        binning = NumericalBinning()

        assert binning.transformation_options is None

    # Handles empty transformation_options without errors
    def test_handling_empty_binning_options(self):
        data = pd.DataFrame({"feature1": [1, 2, 3]})
        binning = NumericalBinning(transformation_options={})
        transformed_data = binning.fit_transform(data)

        pd.testing.assert_frame_equal(transformed_data, data)

    # Correctly sets and retrieves parameters using set_params and get_params
    def test_set_and_get_params(self):
        transformation_options = {
            "column1": ("uniform", 5),
            "column2": ("quantile", 10),
        }
        binning = NumericalBinning(transformation_options=transformation_options)

        params = binning.get_params()
        binning.set_params(transformation_options={"column3": ("kmeans", 8)})

        updated_params = binning.get_params()

        assert params == {
            "transformation_options": transformation_options,
            "track_columns": False,
        }
        assert updated_params == {
            "transformation_options": {"column3": ("kmeans", 8)},
            "track_columns": False,
        }

    # Successfully fits data using specified binning strategies and number of bins
    def test_fit_with_binning_strategies(self):
        transformation_options = {
            "column1": ("uniform", 5),
            "column2": ("quantile", 10),
        }
        X = pd.DataFrame({"column1": [1, 2, 3, None], "column2": [4, 5, None, 7]})

        binning = NumericalBinning(transformation_options=transformation_options)
        binning.fit(X)

        assert len(binning._binners) == 2
        assert "column1" in binning._binners
        assert "column2" in binning._binners
