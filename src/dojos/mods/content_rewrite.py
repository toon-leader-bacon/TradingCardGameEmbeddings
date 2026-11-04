"""Whole-tree rewrites of a card's raw_content, for the content style mods
(content_style_mods.py).

A ContentRewrite (Strategy) says what to do with each key, each scalar
leaf and each list; rewritten_content applies it to every node of the
tree and builds a new tree, never touching the old one. flattened_content
is a different shape of walk (it moves nested keys up to the top level),
so it is its own function rather than a ContentRewrite.

Nested JSON has a data-defined, shallow depth (three levels or fewer in
practice), so these walkers recurse rather than manage an explicit stack,
as src/data_refinement/card_binder/lean_content.py does.
"""

from src.schema.json_types import JsonObject, JsonScalar, JsonValue


class KeyCollisionError(ValueError):
    """Two keys of one object became the same key after a rewrite (e.g.
    "mana_cost" and "manaCost" under camel case), or a flattened path
    equals a key already at the top level. The rewrite would lose a value,
    so it is refused; the mod's best-effort guard passes the card through.
    """


class ContentRewrite:
    """Strategy: per-node rules for rewritten_content. Every rule is the
    identity by default, so a rewrite overrides only the nodes it changes.
    Instances hold their own settings (a key case, a separator) and are
    built per card by a mod, never shared mutable state.
    """

    def key(self, key: str) -> str:
        """The new name for one object key.

        Inputs: key. Output: str. Side effects: none. Exceptions: none.
        """
        return key

    def leaf(self, value: JsonScalar) -> JsonValue:
        """The new value for one scalar (None, bool, int, float, str).

        Inputs: value. Output: JsonValue. Side effects: none.
        Exceptions: none.
        """
        return value

    def list_replacement(self, items: list[JsonValue]) -> JsonValue | None:
        """A replacement for a whole list, or None to keep it a list and
        rewrite its items one by one.

        Inputs: items (the list as it is before its items are rewritten).
        Output: JsonValue, or None to descend into it.
        Side effects: none. Exceptions: none.
        """
        return None


def rewritten_content(content: JsonObject, rewrite: ContentRewrite) -> JsonObject:
    """A new tree with rewrite applied to every key, list and scalar of
    content, at every depth. Key order is kept.

    Inputs: content (a card's raw_content), rewrite.
    Output: JsonObject (new dicts and lists throughout; content unchanged).
    Side effects: none.
    Exceptions: KeyCollisionError if two keys of one object get the same
        new name.

    Example:
        >>> class Upper(ContentRewrite):
        ...     def key(self, key): return key.upper()
        >>> rewritten_content({"a": {"b": 1}}, Upper())
        {'A': {'B': 1}}
    """
    result: JsonObject = {}

    # Rewrite each key and its value, refusing a key that lands twice
    for key, value in content.items():
        new_key = rewrite.key(key)
        if new_key in result:
            raise KeyCollisionError(f"keys collide as {new_key!r}")
        result[new_key] = _rewritten_value(value, rewrite)
    return result


def flattened_content(content: JsonObject, separator: str) -> JsonObject:
    """content with every nested object moved up to the top level, its
    path joined by separator: {"stats": {"hp": 60}} -> {"stats.hp": 60}.
    A list stays a list at its own path, but each object inside it is
    flattened in place ({"attacks": [{"cost": {"fire": 1}}]} ->
    {"attacks": [{"cost.fire": 1}]}). An empty object stays as a value.
    Key order follows a depth-first walk, so a nested object's keys sit
    where the object was.

    Inputs: content, separator (a non-empty str).
    Output: JsonObject (new; content unchanged).
    Side effects: none.
    Exceptions: KeyCollisionError if a joined path equals another key
        (e.g. "stats.hp" already present beside "stats": {"hp": ...}).

    Example:
        >>> flattened_content({"name": "Pikachu", "stats": {"hp": 60}}, ".")
        {'name': 'Pikachu', 'stats.hp': 60}
    """
    result: JsonObject = {}

    # Each top-level entry adds itself, or its whole subtree, under its path
    for key, value in content.items():
        _add_flattened(result, key, value, separator)
    return result


def _rewritten_value(value: JsonValue, rewrite: ContentRewrite) -> JsonValue:
    """rewritten_content for one value of any JSON type: an object is
    rewritten key by key, a list is offered to rewrite.list_replacement then
    descended, a scalar goes to rewrite.leaf.

    Inputs: value, rewrite. Output: JsonValue (new containers).
    Side effects: none.
    Exceptions: KeyCollisionError, as rewritten_content.
    """
    if isinstance(value, dict):
        return rewritten_content(value, rewrite)
    if isinstance(value, list):
        replacement = rewrite.list_replacement(value)
        if replacement is not None:
            return replacement
        return [_rewritten_value(item, rewrite) for item in value]
    return rewrite.leaf(value)


def _add_flattened(
    result: JsonObject, path: str, value: JsonValue, separator: str
) -> None:
    """Add value to result under path, or, for a non-empty object, each
    of its entries under path + separator + key (recursing). A list is
    added under path with its objects flattened in place
    (_flattened_list_items).

    Inputs: result (the flat object being built), path, value, separator.
    Output: none.
    Side effects: adds keys to result.
    Exceptions: KeyCollisionError if a path is already in result.
    """
    # A non-empty object contributes its entries, each under a longer path
    if isinstance(value, dict) and value:
        for key, nested in value.items():
            _add_flattened(result, f"{path}{separator}{key}", nested, separator)
        return

    # Anything else lands under path itself
    if path in result:
        raise KeyCollisionError(f"flattened path {path!r} is already a key")
    if isinstance(value, list):
        value = _flattened_list_items(value, separator)
    elif isinstance(value, dict):
        value = {}  # the empty object, new like every other container
    result[path] = value


def _flattened_list_items(items: list[JsonValue], separator: str) -> list[JsonValue]:
    """A new list with each object item flattened (flattened_content) and
    each nested list handled the same way; scalars unchanged.

    Inputs: items, separator. Output: list[JsonValue].
    Side effects: none. Exceptions: KeyCollisionError, as flattened_content.
    """
    result: list[JsonValue] = []
    for item in items:
        if isinstance(item, dict):
            result.append(flattened_content(item, separator))
        elif isinstance(item, list):
            result.append(_flattened_list_items(item, separator))
        else:
            result.append(item)
    return result
