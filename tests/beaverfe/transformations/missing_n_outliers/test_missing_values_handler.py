import numpy as np
import pandas as pd

from beaverfe.transformations import MissingValuesHandler


class TestMissingValuesHandler:
    # Initialize MissingValuesHandler with valid handling_options and n_neighbors
    def test_initialization_with_valid_params(self):
        handling_options = {"column1": "fill_mean", "column2": "fill_knn"}
        n_neighbors = {"column2": 3}
        handler = MissingValuesHandler(
            transformation_options=handling_options, n_neighbors=n_neighbors
        )

        assert handler.transformation_options == handling_options
        assert handler.n_neighbors == n_neighbors

    # Handling_options is None or empty
    def test_handling_options_none_or_empty(self):
        handler_none = MissingValuesHandler(transformation_options=None)
        handler_empty = MissingValuesHandler(transformation_options={})

        assert handler_none.transformation_options is None
        assert handler_empty.transformation_options == {}

    # Call fit with a DataFrame and verify statistics and imputers are correctly set
    def test_fit_with_dataframe(self):
        # Creating an instance of MissingValuesHandler
        handler = MissingValuesHandler(
            transformation_options={"col1": "mean", "col2": "knn"},
            n_neighbors={"col2": 3},
        )

        # Creating a mock DataFrame
        X = pd.DataFrame({"col1": [1, 2, np.nan], "col2": [3, np.nan, 5]})

        # Calling the fit method
        handler.fit(X)

        # Assertions to verify imputers are correctly set
        assert "col1" in handler._imputers
        assert "col2" in handler._imputers

    # Use transform to fill missing values based on handling_options
    def test_transform_fill_missing_values(self):
        # Creating an instance of MissingValuesHandler
        handler = MissingValuesHandler(
            transformation_options={"col1": "mean", "col2": "knn"},
            n_neighbors={"col2": 3},
        )

        # Creating a mock DataFrame
        X = pd.DataFrame({"col1": [1, 2, np.nan], "col2": [3, np.nan, 5]})

        # Calling the fit method
        handler.fit(X)

        # Calling the transform method
        X_transformed = handler.transform(X)

        # Assertions to verify missing values are correctly filled
        assert X_transformed["col1"].isnull().sum() == 0
        assert X_transformed["col2"].isnull().sum() == 0
