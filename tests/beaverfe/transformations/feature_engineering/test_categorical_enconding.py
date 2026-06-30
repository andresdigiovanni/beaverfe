import pandas as pd

from beaverfe.transformations import CategoricalEncoding


class TestCategoricalEncoding:
    # Initialize CategoricalEncoding with default parameters and verify default behavior
    def test_default_initialization(self):
        encoder = CategoricalEncoding()
        params = encoder.get_params()

        assert params["transformation_options"] == {}
        assert params["ordinal_orders"] is None

    # Fit and transform a DataFrame with various encoding options and verify transformations
    def test_fit_and_transform_with_encoding_options(self):
        X = pd.DataFrame(
            {
                "col1": ["A", "B", "C", "D", "E"],
                "col2": ["A", "B", "C", "D", "E"],
                "col3": ["A", "B", "C", "D", "E"],
                "col4": ["A", "B", "C", "D", "E"],
                "col5": ["A", "B", "C", "D", "E"],
            }
        )
        y = pd.Series([0, 1, 1, 1, 0])

        ce = CategoricalEncoding(
            transformation_options={
                "col1": "label",
                "col2": "ordinal",
                "col3": "onehot",
                "col4": "binary",
                "col5": "target",
            },
            ordinal_orders={"col2": ["C", "D", "E", "A", "B"]},
        )
        ce.fit(X, y)
        transformed_X = ce.transform(X, y)

        # Assertions for the transformations
        expected_X = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [3, 4, 0, 1, 2],
                "col3_1": [1, 0, 0, 0, 0],
                "col3_2": [0, 1, 0, 0, 0],
                "col3_3": [0, 0, 1, 0, 0],
                "col3_4": [0, 0, 0, 1, 0],
                "col3_5": [0, 0, 0, 0, 1],
                "col4_0": [0, 0, 0, 1, 1],
                "col4_1": [0, 1, 1, 0, 0],
                "col4_2": [1, 0, 1, 0, 1],
                "col5": [0.521935, 0.652043, 0.652043, 0.652043, 0.521935],
            }
        )

        pd.testing.assert_frame_equal(transformed_X, expected_X, check_dtype=False)
