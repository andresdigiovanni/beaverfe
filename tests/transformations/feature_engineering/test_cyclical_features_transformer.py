import pandas as pd

from beaverfe.transformations import CyclicalFeaturesTransformer


class TestCyclicalFeaturesTransformer:

    # Transforms data by adding sine and cosine features for specified columns
    def test_transform_adds_sine_and_cosine_features(self):
        data = pd.DataFrame({'angle': [0, 90, 180, 270]})
        transformer = CyclicalFeaturesTransformer(columns_periods={'angle': 360})
        transformed_data = transformer.transform(data)
    
        assert 'angle_sin' in transformed_data.columns
        assert 'angle_cos' in transformed_data.columns
        assert transformed_data['angle_sin'].iloc[0] == 0
        assert transformed_data['angle_cos'].iloc[0] == 1
