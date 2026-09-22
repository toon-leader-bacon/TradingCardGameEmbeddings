"""Source-independent helpers that keep an ingested card's raw_content lean.

A text encoder reads raw_content verbatim, so noise in it (URLs, IDs,
timestamps, empty fields) costs tokens and, once a text is truncated, can
push the useful fields out entirely. Each CardIngestionStage decides which
of its source's keys describe the game object (see "What goes in
raw_content" in card_binder/README.md); the functions here are the parts of
that job that do not depend on the source, so stages share them instead of
each re-implementing them. Which keys to keep, their order, and which rows
are cards stay in each stage.

Every function returns new objects and never mutates its input, because a
GenericCard's raw_content must not be changed in place.

Nested JSON has a data-defined depth (three levels or fewer in practice),
so the walkers below recurse rather than manage an explicit stack.
"""

import re
from typing import Callable, Sequence, Union

JsonValue = Union[
    None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]
]
JsonObject = dict[str, JsonValue]

_URL = re.compile(r"https?://\S+")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_DATE_OR_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(T[\d:.]+Z?)?")
# id, card_id, multiverse_ids, image_uris, thumbnail_url ... but not "grid"
_NOISE_KEY = re.compile(r"(^|_)(ids?|uris?|urls?)$")

# Shorter strings ("Yes", "0", "Red") repeat by coincidence, not by redundancy
_MIN_REPEATED_STRING_LENGTH = 4


def strip_noise(
    content: JsonObject, extra_noise_keys: frozenset[str] = frozenset()
) -> JsonObject:
    """Remove the values and keys that carry no information about the card.

    Removed, at every nesting depth: strings that are a URL, a UUID or an
    ISO date/timestamp (whatever their key); keys named id/*_id/*_ids,
    *_uri/*_uris and *_url/*_urls; keys listed in extra_noise_keys; and
    every value left empty (None, "", [], {}, False) including containers
    that became empty. Zero and True are kept. Whole floats become ints
    (3.0 -> 3).

    Inputs: content (JsonObject), extra_noise_keys (frozenset[str], key
        names to drop wherever they appear, e.g. a source's printing or
        API-envelope fields).
    Output: a new JsonObject.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> strip_noise({"name": "Bolt", "id": "7", "cmc": 1.0, "text": ""})
        {'name': 'Bolt', 'cmc': 1}
    """
    return _strip_dict(content, extra_noise_keys)


def map_strings(content: JsonObject, transform: Callable[[str], str]) -> JsonObject:
    """Apply a text transform to every string value, at every depth.

    For presentation markup in text (color tags, line-break codes) that a
    source's stage wants rewritten or removed. Keys are left alone.

    Inputs: content (JsonObject), transform (Callable[[str], str]).
    Output: a new JsonObject with the same shape.
    Side effects: none (beyond whatever transform does).
    Exceptions: whatever transform raises.

    Example:
        >>> map_strings({"text": "Gain [gold]Block[/gold]"}, lambda s: s.replace("[gold]", ""))
        {'text': 'Gain Block[/gold]'}
    """
    mapped = _map_strings_in(content, transform)
    assert isinstance(mapped, dict)
    return mapped


def drop_repeated_strings(content: JsonObject) -> JsonObject:
    """Within each object, keep the first of any repeated string value.

    Only strings of at least four characters that are not all digits count,
    since short strings and numbers repeat by coincidence. Later keys whose
    value equals an earlier key's value in the same object are dropped, so
    put the preferred key first.

    Inputs: content (JsonObject).
    Output: a new JsonObject.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> drop_repeated_strings({"name": "Bolt", "true_name": "Bolt", "cost": "1"})
        {'name': 'Bolt', 'cost': '1'}
    """
    seen: set[str] = set()
    kept: JsonObject = {}
    for key, value in content.items():
        if isinstance(value, str) and _can_repeat_by_redundancy(value):
            if value in seen:
                continue
            seen.add(value)
        kept[key] = _drop_repeated_strings_in(value)
    return kept


def order_keys(
    content: JsonObject,
    leading: Sequence[str] = (),
    trailing: Sequence[str] = (),
) -> JsonObject:
    """Put short, identifying keys first and long free text last.

    If the encoder truncates, only the tail is lost. Keys not named in
    leading or trailing keep their relative order in the middle; a named key
    the content lacks is skipped.

    Inputs: content (JsonObject), leading and trailing (key names, in the
        order wanted).
    Output: a new JsonObject with the same items.
    Side effects: none.
    Exceptions: ValueError if a key is in both leading and trailing.

    Example:
        >>> order_keys({"text": "t", "extra": 1, "name": "n"}, ["name"], ["text"])
        {'name': 'n', 'extra': 1, 'text': 't'}
    """
    overlap = set(leading) & set(trailing)
    if overlap:
        raise ValueError(f"keys cannot be both leading and trailing: {sorted(overlap)}")
    head = {key: content[key] for key in leading if key in content}
    tail = {key: content[key] for key in trailing if key in content}
    middle = {k: v for k, v in content.items() if k not in head and k not in tail}
    return {**head, **middle, **tail}


def _strip_dict(content: JsonObject, extra_noise_keys: frozenset[str]) -> JsonObject:
    stripped: JsonObject = {}
    for key, value in content.items():
        if key in extra_noise_keys or _NOISE_KEY.search(key):
            continue
        cleaned = _strip_value(value, extra_noise_keys)
        if not _is_empty(cleaned):
            stripped[key] = cleaned
    return stripped


def _strip_value(value: JsonValue, extra_noise_keys: frozenset[str]) -> JsonValue:
    """The cleaned value; None (which the caller drops) for a noise string."""
    if isinstance(value, dict):
        return _strip_dict(value, extra_noise_keys)
    if isinstance(value, list):
        items = [_strip_value(item, extra_noise_keys) for item in value]
        return [item for item in items if not _is_empty(item)]
    if isinstance(value, str):
        return None if _is_noise_string(value) else value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _is_noise_string(text: str) -> bool:
    return bool(
        _URL.fullmatch(text)
        or _UUID.fullmatch(text)
        or _DATE_OR_TIMESTAMP.fullmatch(text)
    )


def _is_empty(value: JsonValue) -> bool:
    """None, "", [], {} and False; zero is not empty."""
    return value is None or value is False or value == "" or value == [] or value == {}


def _can_repeat_by_redundancy(text: str) -> bool:
    return len(text) >= _MIN_REPEATED_STRING_LENGTH and not text.isdigit()


def _drop_repeated_strings_in(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return drop_repeated_strings(value)
    if isinstance(value, list):
        return [_drop_repeated_strings_in(item) for item in value]
    return value


def _map_strings_in(value: JsonValue, transform: Callable[[str], str]) -> JsonValue:
    if isinstance(value, dict):
        return {key: _map_strings_in(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings_in(item, transform) for item in value]
    if isinstance(value, str):
        return transform(value)
    return value
