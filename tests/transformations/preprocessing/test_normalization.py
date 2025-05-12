
import pandas as pd
import pytest

from beaverfe.transformations import Normalization


class TestNormalization:

    # Initializes with default transformation options when none are provided
    def test_initialization_with_default_options(self):
        normalization = Normalization()

        assert normalization.transformation_options == {}

    # Handles empty transformation_options gracefully without errors
    def test_empty_transformation_options(self):
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        
        normalization = Normalization(transformation_options={})
        transformed_data = normalization.fit_transform(data)

        pd.testing.assert_frame_equal(data, transformed_data)

    # Correctly sets and retrieves parameters using set_params and get_params
    def test_set_and_get_params(self):
        # Arrange
        normalization = Normalization()
        params = {"transformation_options": {"col1": "l1", "col2": "l2"}}

        # Act
        normalization.set_params(**params)
        retrieved_params = normalization.get_params()

        # Assert
        assert retrieved_params == params

    # Successfully fits transformers for specified columns with 'l1' or 'l2' normalization
    def test_fit_transformers(self):
        # Arrange
        normalization = Normalization(transformation_options={"col1": "l1", "col2": "l2"})
        X = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        # Act
        normalization.fit(X)

        # Assert
        assert len(normalization._transformers) == 2