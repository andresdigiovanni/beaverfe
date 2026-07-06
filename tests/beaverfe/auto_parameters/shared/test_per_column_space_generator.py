import pandas as pd

from beaverfe.auto_parameters.shared.space_generator import PerColumnSpaceGenerator


class _AlwaysMenuGenerator(PerColumnSpaceGenerator):
    """Mirrors OutliersSpaceGenerator/ScaleTransformation etc: always offers
    a menu, even with zero extra options beyond "none"."""

    prefix = "always"

    def __init__(self, options_by_column):
        self._options_by_column = options_by_column

    def _options_for_column(self, X, col):
        return self._options_by_column.get(col, [])


class _ConditionalMenuGenerator(PerColumnSpaceGenerator):
    """Mirrors NonLinearTransformationSpaceGenerator: omits columns
    entirely when there's nothing to offer."""

    prefix = "conditional"

    def __init__(self, eligible_columns):
        self._eligible_columns = eligible_columns

    def _options_for_column(self, X, col):
        return ["some_option"] if col in self._eligible_columns else None


class TestPerColumnSpaceGenerator:
    def test_should_still_offer_none_only_menu_when_no_extra_options(self):
        # Regression test: an empty options list must not be treated the
        # same as "omit this column" for generators that always offer a
        # menu (even if it's just "none").
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        gen = _AlwaysMenuGenerator(options_by_column={"a": []})

        space = gen.get_search_space(X, None)

        assert space == {"always_a": ["none"]}

    def test_should_omit_column_when_options_for_column_returns_none(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        gen = _ConditionalMenuGenerator(eligible_columns={"a"})

        space = gen.get_search_space(X, None)

        assert "conditional_a" in space
        assert "conditional_b" not in space

    def test_should_return_empty_space_when_no_numeric_columns(self):
        X = pd.DataFrame({"a": ["x", "y", "z"]})
        gen = _AlwaysMenuGenerator(options_by_column={})

        space = gen.get_search_space(X, None)

        assert space == {}
