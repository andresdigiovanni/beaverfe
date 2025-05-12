import pandas as pd

from beaverfe.transformations import DateTimeTransformer


class TestDateTimeTransformer:

    # Transform datetime columns into separate year, month, day, weekday, hour, minute, and second columns
    def test_transform_datetime_columns(self):
        data = pd.DataFrame({
            'datetime_col': pd.to_datetime(['2023-01-01 12:34:56', '2023-06-15 23:45:01'])
        })
        transformer = DateTimeTransformer(datetime_columns=['datetime_col'])
        transformed_data = transformer.transform(data)
    
        expected_columns = ['datetime_col_year', 'datetime_col_month', 'datetime_col_day', 
                            'datetime_col_weekday', 'datetime_col_hour', 'datetime_col_minute', 
                            'datetime_col_second']
    
        assert all(col in transformed_data.columns for col in expected_columns)
        assert transformed_data['datetime_col_year'].tolist() == [2023, 2023]
        assert transformed_data['datetime_col_month'].tolist() == [1, 6]
        assert transformed_data['datetime_col_day'].tolist() == [1, 15]
        assert transformed_data['datetime_col_weekday'].tolist() == [6, 3]
        assert transformed_data['datetime_col_hour'].tolist() == [12, 23]
        assert transformed_data['datetime_col_minute'].tolist() == [34, 45]
        assert transformed_data['datetime_col_second'].tolist() == [56, 1]
