import pytest

from src.schema.json_checks import (
    optional_str,
    require_int,
    require_nonempty_str,
    require_object,
    require_object_list,
    require_str,
)


class TestRequireInt:
    def test_an_int_passes(self) -> None:
        assert require_int(3) == 3

    @pytest.mark.parametrize("value", [True, 1.0, "1", None])
    def test_other_values_raise(self, value: object) -> None:
        with pytest.raises(TypeError):
            require_int(value)  # type: ignore[arg-type]


class TestRequireStr:
    def test_a_str_passes(self) -> None:
        assert require_str("") == ""

    @pytest.mark.parametrize("value", [1, None, ["a"]])
    def test_other_values_raise(self, value: object) -> None:
        with pytest.raises(TypeError):
            require_str(value)  # type: ignore[arg-type]


class TestRequireObject:
    def test_an_object_passes(self) -> None:
        assert require_object({"a": 1}) == {"a": 1}

    @pytest.mark.parametrize("value", [[], "a", None])
    def test_other_values_raise(self, value: object) -> None:
        with pytest.raises(TypeError):
            require_object(value)  # type: ignore[arg-type]


class TestRequireObjectList:
    def test_a_list_of_objects_passes(self) -> None:
        assert require_object_list([{"a": 1}, {}]) == [{"a": 1}, {}]
        assert require_object_list([]) == []

    @pytest.mark.parametrize("value", [{"a": 1}, [1], [{"a": 1}, "b"]])
    def test_other_values_raise(self, value: object) -> None:
        with pytest.raises(TypeError):
            require_object_list(value)  # type: ignore[arg-type]


class TestRequireNonemptyStr:
    def test_a_nonempty_str_passes(self) -> None:
        assert require_nonempty_str("a") == "a"

    @pytest.mark.parametrize("value", ["", 1, None])
    def test_other_values_raise(self, value: object) -> None:
        with pytest.raises(TypeError):
            require_nonempty_str(value)  # type: ignore[arg-type]


class TestOptionalStr:
    def test_a_str_or_null_passes(self) -> None:
        assert optional_str("a") == "a"
        assert optional_str(None) is None

    def test_other_values_raise(self) -> None:
        with pytest.raises(TypeError):
            optional_str(1)
