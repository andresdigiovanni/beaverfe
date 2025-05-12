
import pandas as pd
import pytest

from beaverfe.transformations import OutliersHandler


class TestOutliersHandler:

    # Correctly initializes with default and custom parameters
    def test_initialization_with_parameters(self):
        handling_options = {'feature1': ('median', 'lof')}
        thresholds = {'feature1': 1.5}
        lof_params = {'feature1': {'n_neighbors': 20}}
        iforest_params = {'feature1': {'contamination': 0.1}}
    
        handler = OutliersHandler(
            handling_options=handling_options,
            thresholds=thresholds,
            lof_params=lof_params,
            iforest_params=iforest_params
        )
    
        assert handler.handling_options == handling_options
        assert handler.thresholds == thresholds
        assert handler.lof_params == lof_params
        assert handler.iforest_params == iforest_params

    # Handles empty or null dataframes gracefully
    def test_handle_empty_dataframe(self):
        handler = OutliersHandler(
            handling_options={},
            thresholds={},
            lof_params={},
            iforest_params={}
        )
    
        empty_df = pd.DataFrame()
        transformed_df = handler.fit_transform(empty_df)
    
        assert transformed_df.empty

    # Successfully fits data using specified outlier detection methods
    def test_successfully_fits_data(self):
        # Setup
        X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        y = None

        # Given
        handling_options = {
            'A': ('median', 'lof'),
            'B': ('median', 'iforest')
        }
        thresholds = {'A': 1.5, 'B': 2.0}
        lof_params = {'A': {'n_neighbors': 5}, 'B': {'n_neighbors': 10}}
        iforest_params = {'A': {'contamination': 0.1}, 'B': {'contamination': 0.2}}

        outliers_handler = OutliersHandler(
            handling_options=handling_options,
            thresholds=thresholds,
            lof_params=lof_params,
            iforest_params=iforest_params
        )

        # When
        outliers_handler.fit(X, y)

        # Then
        assert 'A' in outliers_handler._lof_results
        assert 'B' in outliers_handler._iforest_results


    # Accurately transforms data based on fitted outlier detection
    def test_accurately_transforms_data(self):
        # Setup
        X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 60]
        })
        y = None

        # Given
        handling_options = {
            'A': ('median', 'lof'),
            'B': ('median', 'iforest')
        }
        thresholds = {}
        lof_params = {'A': {'n_neighbors': 5}}
        iforest_params = { 'B': {'contamination': 0.1}}

        outliers_handler = OutliersHandler(
            handling_options=handling_options,
            thresholds=thresholds,
            lof_params=lof_params,
            iforest_params=iforest_params
        )
        outliers_handler.fit(X, y)

        # When
        X_transformed = outliers_handler.transform(X, y)

        # Then
        X_expected = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 30.0]
        })

        assert X_transformed['A'].equals(X_expected['A'])
        assert X_transformed['B'].equals(X_expected['B'])
