import pandas as pd

from beaverfe.transformations import NonLinearTransformation


class TestNonLinearTransformation:

    # Initialize with valid transformation options and apply transformations correctly
    def test_valid_transformation_options(self):
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
    
        transformation_options = {
            'feature1': 'log',
            'feature2': 'yeo_johnson'
        }
    
        transformer = NonLinearTransformation(transformation_options=transformation_options)
        transformed_data = transformer.fit_transform(data)
    
        assert not transformed_data.isnull().values.any(), "Transformed data should not contain NaN values"

    # Handle empty transformation_options gracefully without errors
    def test_empty_transformation_options(self):    
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
    
        transformer = NonLinearTransformation(transformation_options={})
        transformed_data = transformer.fit_transform(data)
    
        pd.testing.assert_frame_equal(data, transformed_data, check_dtype=False)