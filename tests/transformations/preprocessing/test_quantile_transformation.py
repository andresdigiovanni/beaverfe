
import numpy as np
import pandas as pd
import pytest

from beaverfe.transformations import QuantileTransformation


class TestQuantileTransformation:

    # Initialize QuantileTransformation with default options and verify no transformers are set
    def test_initialization_with_default_options(self):
        qt = QuantileTransformation()

        assert qt.transformation_options == {}
        assert qt._transformers == {}

    # Pass an empty DataFrame to fit and ensure no errors occur
    def test_fit_with_empty_dataframe(self):
        qt = QuantileTransformation()
        empty_df = pd.DataFrame()

        qt.fit(empty_df)
        
        assert qt._transformers == {}

    # Set transformation options and ensure fit method initializes transformers correctly
    def test_set_transformation_options(self):
        # Setup
        transformation_options = {"feature1": "uniform", "feature2": "normal"}
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        # Exercise
        qt = QuantileTransformation(transformation_options)
        qt.fit(data)

        # Verify
        assert len(qt._transformers) == 2
        assert "feature1" in qt._transformers
        assert "feature2" in qt._transformers

    # Transform data using fit_transform and verify output matches expected quantile transformation
    def test_transform_data(self):
        # Setup
        transformation_options = {"feature1": "uniform", "feature2": "normal"}
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        # Exercise
        qt = QuantileTransformation(transformation_options)
        X_transformed = qt.fit_transform(data)

        # Verify
        X_expected = pd.DataFrame({
            'feature1': [0.00, 0.25, 0.50, 0.75, 1.00],
            'feature2': [-5.199338, -0.674490, 0.000000, 0.674490, 5.199338]
        })

        assert np.allclose(X_transformed["feature1"], X_expected["feature1"], rtol=1e-3)
        assert np.allclose(X_transformed["feature2"], X_expected["feature2"], rtol=1e-3)
