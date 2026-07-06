import numpy as np
import pandas as pd
import pytest

from beaverfe.transformations import SplineTransformation


@pytest.fixture
def X():
    rng = np.random.default_rng(42)
    return pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})


class TestSplineTransformation:
    def test_should_add_spline_basis_columns(self, X):
        transformer = SplineTransformation(
            transformation_options={"a": {"n_knots": 3, "degree": 2}}
        )

        result = transformer.fit_transform(X)

        spline_columns = [c for c in result.columns if c.startswith("a__spline_")]
        assert len(spline_columns) > 0

    def test_should_preserve_original_columns(self, X):
        transformer = SplineTransformation(
            transformation_options={"a": {"n_knots": 3, "degree": 2}}
        )

        result = transformer.fit_transform(X)

        assert "a" in result.columns
        assert "b" in result.columns

    def test_should_only_transform_requested_columns(self, X):
        transformer = SplineTransformation(
            transformation_options={"a": {"n_knots": 3, "degree": 2}}
        )

        result = transformer.fit_transform(X)

        assert not any(c.startswith("b__spline_") for c in result.columns)

    def test_should_handle_empty_transformation_options(self, X):
        transformer = SplineTransformation(transformation_options={})

        result = transformer.fit_transform(X)

        pd.testing.assert_frame_equal(result, X)

    def test_should_track_columns_when_enabled(self, X):
        transformer = SplineTransformation(
            transformation_options={"a": {"n_knots": 3, "degree": 2}},
            track_columns=True,
        )

        transformer.fit_transform(X)

        assert all(v == ["a"] for v in transformer.tracked_columns.values())
        assert len(transformer.tracked_columns) > 0

    def test_should_round_trip_get_set_params(self):
        transformer = SplineTransformation(
            transformation_options={"a": {"n_knots": 5, "degree": 3}}
        )

        params = transformer.get_params()
        transformer.set_params(transformation_options={"b": {"n_knots": 3}})

        assert params == {
            "transformation_options": {"a": {"n_knots": 5, "degree": 3}},
            "track_columns": False,
        }
        assert transformer.transformation_options == {"b": {"n_knots": 3}}
