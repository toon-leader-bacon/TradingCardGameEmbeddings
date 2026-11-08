"""Checks that narrow a parsed JSON value to the type a reader expects,
raising TypeError otherwise. Plain functions, for parsers that read raw
JSON (crawl rows, files a downloader wrote).
"""

from src.schema.json_types import JsonObject, JsonValue


def require_int(value: JsonValue) -> int:
    """value if it is an int (not a bool).

    Inputs: value. Output: int. Side effects: none.
    Exceptions: TypeError otherwise.

    Example:
        >>> require_int(3)
        3
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an int, got {value!r}")
    return value


def require_str(value: JsonValue) -> str:
    """value if it is a str.

    Inputs: value. Output: str. Side effects: none.
    Exceptions: TypeError otherwise.

    Example:
        >>> require_str("325-Burn")
        '325-Burn'
    """
    if not isinstance(value, str):
        raise TypeError(f"expected a str, got {value!r}")
    return value


def optional_str(value: JsonValue) -> str | None:
    """value if it is a str, None if it is null.

    Inputs: value. Output: str or None. Side effects: none.
    Exceptions: TypeError otherwise.

    Example:
        >>> optional_str(None) is None
        True
    """
    return None if value is None else require_str(value)


def require_nonempty_str(value: JsonValue) -> str:
    """value if it is a non-empty str.

    Inputs: value. Output: str. Side effects: none.
    Exceptions: TypeError otherwise.

    Example:
        >>> require_nonempty_str("Lightning Bolt")
        'Lightning Bolt'
    """
    if not isinstance(value, str) or not value:
        raise TypeError(f"expected a non-empty str, got {value!r}")
    return value


def require_object(value: JsonValue) -> JsonObject:
    """value if it is a JSON object.

    Inputs: value. Output: JsonObject. Side effects: none.
    Exceptions: TypeError otherwise.

    Example:
        >>> require_object({"id": 1})
        {'id': 1}
    """
    if not isinstance(value, dict):
        raise TypeError(f"expected an object, got {value!r}")
    return value


def require_object_list(value: JsonValue) -> list[JsonObject]:
    """value if it is a list of JSON objects.

    Inputs: value. Output: list[JsonObject]. Side effects: none.
    Exceptions: TypeError if value is not a list or an item is not an
        object.

    Example:
        >>> require_object_list([{"id": 1}])
        [{'id': 1}]
    """
    if not isinstance(value, list):
        raise TypeError(f"expected a list, got {value!r}")
    return [require_object(item) for item in value]
