

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from beaverfe.transformations import ScaleTransformation


class TestScaleTransformation:

    # Initialize ScaleTransformation with default options and verify no transformers are set
    def test_initialize_with_default_options(self):
        scale_transformation = ScaleTransformation()

        assert scale_transformation.transformation_options == {}
        assert scale_transformation._transformers == {}

    # Initialize with empty transformation options and verify no errors occur during fit
    def test_fit_with_empty_transformation_options(self):
        scale_transformation = ScaleTransformation(transformation_options={})
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        scale_transformation.fit(df)

        assert scale_transformation._transformers == {}

    # Set transformation options and ensure correct scalers are initialized during fit
    def test_set_transformation_options(self):
        # Given
        transformation_options = {"feature1": "min_max", "feature2": "standard"}
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        # When
        scaler = ScaleTransformation(transformation_options)
        X_transformed = scaler.fit_transform(X)

        # Then
        X_expected= pd.DataFrame({
            'feature1': [0.00, 0.25, 0.50, 0.75, 1.00],
            'feature2': [-1.414214, -0.707107, 0.000000, 0.707107, 1.414214]
        })

        assert "feature1" in scaler._transformers
        assert "feature2" in scaler._transformers
        assert isinstance(scaler._transformers["feature1"], MinMaxScaler)
        assert isinstance(scaler._transformers["feature2"], StandardScaler)
        assert np.allclose(X_transformed["feature1"], X_expected["feature1"], rtol=1e-3)
