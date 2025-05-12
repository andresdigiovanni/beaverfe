import pandas as pd

from beaverfe.transformations import MathematicalOperations


class TestMathematicalOperations:

    # Initialize MathematicalOperations with default operations_options and transform data
    def test_default_operations_options(self):
        # Create a sample DataFrame
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
    
        # Initialize with default operations_options
        transformer = MathematicalOperations()
    
        # Transform the data
        transformed_data = transformer.transform(data)
    
        # Assert that the transformed data is the same as input data
        pd.testing.assert_frame_equal(transformed_data, data)

    # Handle division by zero gracefully without raising errors
    def test_division_by_zero_handling(self):
        # Create a sample DataFrame with a zero value
        data = pd.DataFrame({
            'A': [1, 2, 0],
            'B': [4, 0, 6]
        })
    
        # Initialize with division operation
        transformer = MathematicalOperations(operations_options=[('A', 'B', 'divide')])
    
        # Transform the data
        transformed_data = transformer.transform(data)
    
        # Check that division by zero results in zero (as per fillna(0))
        expected_data = data.copy()
        expected_data['A__divide__B'] = [0.25, 0.0, 0.0]
    
        pd.testing.assert_frame_equal(transformed_data, expected_data)