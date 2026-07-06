import pytest

from beaverfe.transformations import MissingValuesHandler
from beaverfe.utils.get_transformer import get_transformer


class TestGetTransformer:
    def test_should_raise_value_error_when_name_is_unknown(self) -> None:
        with pytest.raises(
            ValueError, match="Unknown transformer: NonExistentTransformer"
        ):
            get_transformer("NonExistentTransformer", {})

    def test_should_raise_value_error_when_name_is_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Unknown transformer: "):
            get_transformer("", {})

    def test_should_raise_value_error_when_name_is_none(self) -> None:
        with pytest.raises(ValueError, match="Unknown transformer: None"):
            get_transformer(None, {})  # type: ignore[arg-type]

    def test_should_return_missing_values_handler_when_name_is_known(self) -> None:
        result = get_transformer("MissingValuesHandler", {"transformation_options": {}})

        assert isinstance(result, MissingValuesHandler)
