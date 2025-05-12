

import pandas as pd
import pytest

from beaverfe.transformations import ColumnSelection


class TestColumnSelection:

    # Initialize ColumnSelection with a list of column names
    def test_initialize_with_column_names(self):
        columns = ['col1', 'col2', 'col3']
        column_selector = ColumnSelection(columns=columns)
        assert column_selector.columns == columns

    # Initialize ColumnSelection with an empty list of columns
    def test_initialize_with_empty_columns(self):
        columns = []
        column_selector = ColumnSelection(columns=columns)
        assert column_selector.columns == columns

    # Transform a DataFrame to select specified columns
    def test_transform_select_specified_columns(self):
        # Arrange
        data = {
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        }
        df = pd.DataFrame(data)
        columns_to_select = ['A', 'C']
        expected_result = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})

        # Act
        column_selector = ColumnSelection(columns=columns_to_select)
        transformed_df = column_selector.transform(df)

        # Assert
        pd.testing.assert_frame_equal(transformed_df, expected_result)

    # Use fit_transform to fit and transform data in one step
    def test_fit_transform_one_step(self):
        # Arrange
        data = {
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        }
        df = pd.DataFrame(data)
        columns_to_select = ['A', 'C']
        expected_result = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})

        # Act
        column_selector = ColumnSelection(columns=columns_to_select)
        transformed_df = column_selector.fit_transform(df)

        # Assert
        pd.testing.assert_frame_equal(transformed_df, expected_result)