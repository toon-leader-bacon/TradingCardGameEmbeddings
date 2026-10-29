"""Type aliases for the JSON a card's raw_content holds (GenericCard types
it as a plain `dict`). Kept free of heavy imports so ingestion code and
training code can both use them.
"""

from typing import TypeAlias, Union

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = Union[
    None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]
]
JsonObject: TypeAlias = dict[str, JsonValue]
