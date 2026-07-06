import pandas as pd

from beaverfe.transformations import MissingValuesIndicator


class TestMissingValuesIndicator:
    def test_should_mark_missing_values_as_one(self):
        X = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
        indicator = MissingValuesIndicator(features=["a"])

        result = indicator.fit_transform(X)

        assert result["a__is_missing"].tolist() == [0, 1, 0]

    def test_should_preserve_original_columns(self):
        X = pd.DataFrame({"a": [1.0, None, 3.0]})
        indicator = MissingValuesIndicator(features=["a"])

        result = indicator.fit_transform(X)

        assert "a" in result.columns
        pd.testing.assert_series_equal(result["a"], X["a"])

    def test_should_create_one_indicator_column_per_feature(self):
        X = pd.DataFrame(
            {"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0], "c": [1.0, 2.0, 3.0]}
        )
        indicator = MissingValuesIndicator(features=["a", "b"])

        result = indicator.fit_transform(X)

        assert "a__is_missing" in result.columns
        assert "b__is_missing" in result.columns
        assert "c__is_missing" not in result.columns

    def test_should_track_columns_when_enabled(self):
        X = pd.DataFrame({"a": [1.0, None, 3.0]})
        indicator = MissingValuesIndicator(features=["a"], track_columns=True)

        indicator.fit_transform(X)

        assert indicator.tracked_columns == {"a__is_missing": ["a"]}

    def test_should_round_trip_get_set_params(self):
        indicator = MissingValuesIndicator(features=["a", "b"])

        params = indicator.get_params()
        indicator.set_params(features=["c"])

        assert params == {"features": ["a", "b"], "track_columns": False}
        assert indicator.features == ["c"]
