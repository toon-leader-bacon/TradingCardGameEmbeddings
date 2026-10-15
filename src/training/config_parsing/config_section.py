"""Typed, key-checked reading of a run config's raw YAML mapping.

ConfigSection reads one mapping and remembers which keys it read, so a
typo fails at startup; the functions here are the scalar checks and
construction helpers every section parser shares (run_config.py and the
other config_parsing/ modules).
"""

import dataclasses
import math
from typing import Any, Callable, Literal, Mapping, TypeVar, cast, get_origin

_T = TypeVar("_T")


class ConfigSection:
    """Typed, key-checked reading of one mapping in the raw document.

    Every getter records the key it read, so reject_unread_keys() can flag
    typos; nested sections come back as ConfigSections carrying their
    dotted location, so errors name where they are. Each parser calls
    reject_unread_keys() on the sections it reads. bool is never accepted
    where int or float is expected (bool subclasses int in Python); an int
    is widened where float is expected, and so is a numeric string,
    because PyYAML reads "1e-3" (no decimal point) as a string.
    """

    def __init__(self, raw: Any, where: str) -> None:
        """Inputs: raw (should be a mapping), where (its dotted location,
        e.g. "phases.0"). Output: none. Side effects: none.
        Exceptions: ValueError if raw is not a mapping with str keys."""
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError(f"{where} must be a mapping with string keys")
        self._raw: dict[str, Any] = raw
        self._where = where
        self._read_keys: set[str] = set()

    @property
    def where(self) -> str:
        """This section's dotted location. Side effects: none.
        Exceptions: none."""
        return self._where

    def location_of(self, key: str) -> str:
        """The dotted location of key, for error messages.
        Inputs: key. Output: str. Side effects: none. Exceptions: none."""
        return f"{self._where}.{key}"

    def required(self, key: str, expected_type: type[_T]) -> _T:
        """The scalar at key, checked against expected_type.

        Inputs: key, expected_type (int, float, str or bool).
        Output: the value (an int or numeric string widened to float for
            float).
        Side effects: records key as read.
        Exceptions: ValueError if missing or wrongly typed.
        """
        return checked_scalar(self._take(key), expected_type, self.location_of(key))

    def optional(self, key: str, expected_type: type[_T], default: _T) -> _T:
        """As required(), but default when key is absent.

        Inputs: key, expected_type, default. Output: the value or default.
        Side effects: records key as read.
        Exceptions: ValueError if present and wrongly typed.
        """
        if key not in self._raw:
            return default
        return self.required(key, expected_type)

    def required_str_list(self, key: str) -> tuple[str, ...]:
        """The list of strings at key.

        Inputs: key. Output: tuple[str, ...].
        Side effects: records key as read.
        Exceptions: ValueError if missing, not a list, or any item is not
            a str.
        """
        items = self._take_list(key)
        return tuple(
            checked_scalar(item, str, f"{self.location_of(key)}.{index}")
            for index, item in enumerate(items)
        )

    def optional_str_list(self, key: str) -> tuple[str, ...]:
        """As required_str_list(), but () when key is absent.

        Inputs: key. Output: tuple[str, ...].
        Side effects: records key as read.
        Exceptions: ValueError if present and malformed.
        """
        if key not in self._raw:
            return ()
        return self.required_str_list(key)

    def required_list(self, key: str) -> list[Any]:
        """The raw list at key; its items are the caller's to check.

        Inputs: key. Output: list[Any].
        Side effects: records key as read.
        Exceptions: ValueError if missing or not a list.
        """
        return self._take_list(key)

    def required_section(self, key: str) -> "ConfigSection":
        """The nested mapping at key, located at "<where>.<key>".

        Inputs: key. Output: ConfigSection.
        Side effects: records key as read.
        Exceptions: ValueError if missing or not a mapping.
        """
        return ConfigSection(self._take(key), self.location_of(key))

    def optional_section(self, key: str) -> "ConfigSection":
        """As required_section(), but an empty section when key is absent.

        Inputs: key. Output: ConfigSection.
        Side effects: records key as read.
        Exceptions: ValueError if present and not a mapping.
        """
        if key not in self._raw:
            return ConfigSection({}, self.location_of(key))
        return self.required_section(key)

    def required_section_list(self, key: str) -> list["ConfigSection"]:
        """The list of mappings at key, each located at "<where>.<key>.<i>".

        Inputs: key. Output: list[ConfigSection].
        Side effects: records key as read.
        Exceptions: ValueError if missing, not a list, or any item is not
            a mapping.
        """
        items = self._take_list(key)
        return [
            ConfigSection(item, f"{self.location_of(key)}.{index}")
            for index, item in enumerate(items)
        ]

    def keys(self) -> list[str]:
        """Every key present, in file order (for sections whose keys are
        data, like `mods:`). Does not count them as read.
        Inputs: none. Output: list[str]. Side effects: none.
        Exceptions: none."""
        return list(self._raw)

    def has(self, key: str) -> bool:
        """Whether key is present. Inputs: key. Output: bool.
        Side effects: none (does not count as reading). Exceptions: none."""
        return key in self._raw

    def reject_unread_keys(self) -> None:
        """Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming (with location) every key never read."""
        unread = sorted(set(self._raw) - self._read_keys)
        if unread:
            locations = [self.location_of(key) for key in unread]
            raise ValueError(f"unknown config keys {locations}")

    def _take(self, key: str) -> Any:
        """The raw value at key. Side effects: records key as read.
        Exceptions: ValueError if missing."""
        if key not in self._raw:
            raise ValueError(f"{self.location_of(key)} is required")
        self._read_keys.add(key)
        return self._raw[key]

    def _take_list(self, key: str) -> list[Any]:
        """The raw list at key. Side effects: records key as read.
        Exceptions: ValueError if missing or not a list."""
        value = self._take(key)
        if not isinstance(value, list):
            raise ValueError(f"{self.location_of(key)} must be a list")
        return value


def parse_by_kind(
    section: ConfigSection, parsers: Mapping[str, Callable[[ConfigSection], _T]]
) -> _T:
    """A `{kind: ..., <its keys>}` section -> the spec its kind's parser
    returns. Shared by mods, pooling and enrichment: a kind -> parser
    table, not an if-chain.

    Inputs: section, parsers (kind name -> parser of the remaining keys).
    Output: the parser's spec.
    Side effects: none.
    Exceptions: ValueError for an unknown kind, a malformed entry, or an
        unknown key (reject_unread_keys).
    """
    kind = section.required("kind", str)
    parser = parsers.get(kind)
    if parser is None:
        raise ValueError(
            f"{section.location_of('kind')} is {kind!r}, want one of {sorted(parsers)}"
        )
    result = construct_at(section.where, lambda: parser(section))
    section.reject_unread_keys()
    return result


def build_flat_dataclass(cls: type[_T], section: ConfigSection) -> _T:
    """Construct a frozen dataclass whose fields are all scalars
    (SaturationSpec, FaultPolicy, HardwareLimits) from a flat section.

    Reads each field of cls that the section has, typed by the field's
    annotation (int, float, str; a Literal alias such as Precision is read
    as str and left to cls's own validation); absent fields take cls's
    defaults. Unknown keys raise.

    Inputs: cls (a dataclass type), section.
    Output: an instance of cls.
    Side effects: none.
    Exceptions: ValueError for an unknown or missing field, a wrongly typed
        value, or whatever cls's own validation raises.
    """
    values: dict[str, Any] = {}
    for dataclass_field in dataclasses.fields(cast(Any, cls)):
        has_default = (
            dataclass_field.default is not dataclasses.MISSING
            or dataclass_field.default_factory is not dataclasses.MISSING
        )
        # Absent with a default: leave it to cls
        if not section.has(dataclass_field.name) and has_default:
            continue
        values[dataclass_field.name] = section.required(
            dataclass_field.name, _scalar_type_of(dataclass_field)
        )
    section.reject_unread_keys()
    return construct_at(section.where, lambda: cls(**values))


def _scalar_type_of(field: "dataclasses.Field[Any]") -> type:
    """The scalar type a flat dataclass field is read as: its annotation,
    or str for a Literal.

    Inputs: field. Output: int, float, str or bool.
    Side effects: none.
    Exceptions: TypeError for any other annotation (a programming error:
        the dataclass is not flat).
    """
    if get_origin(field.type) is Literal:
        return str
    if field.type in (int, float, str, bool):
        return cast(type, field.type)
    raise TypeError(f"field {field.name!r} is not a scalar: {field.type!r}")


def checked_scalar(value: Any, expected_type: type[_T], where: str) -> _T:
    """value, checked against expected_type (see ConfigSection's rules).

    Inputs: value, expected_type, where (for the message).
    Output: value as expected_type.
    Side effects: none.
    Exceptions: ValueError if value does not fit expected_type.
    """
    # bool subclasses int, so a stray `true` would pass as 1
    if isinstance(value, bool) and expected_type is not bool:
        raise ValueError(f"{where} must be {expected_type.__name__}, got a bool")
    if expected_type is float and isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{where} must be a number, got {value!r}") from error
        if not math.isfinite(number):
            raise ValueError(f"{where} must be finite, got {value!r}")
        return cast(_T, number)
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{where} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def construct_at(where: str, build: Callable[[], _T]) -> _T:
    """build(), with any ValueError it raises prefixed by where.

    Inputs: where (dotted location), build (a constructor call).
    Output: build's result.
    Side effects: whatever build does (none for the frozen types).
    Exceptions: ValueError("<where>: <message>").
    """
    try:
        return build()
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
