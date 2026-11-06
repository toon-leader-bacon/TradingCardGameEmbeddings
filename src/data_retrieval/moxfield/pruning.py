"""Unwrapping one Moxfield deck's JSON (~47 KB for a 100-card deck:
prices, store links, card text, legalities) by pruning: kept keys keep
the API's own names and nesting, and nothing is interpreted. Owners and
authors (user names, ids), prices, store links, card text and token
maps are dropped.

Payload shape (api2.moxfield.com/v3/decks/all/<publicId>, 2026-11-05):
{"publicId", "name", "format", ..., "boards": {"mainboard": {"count",
"cards": {key: {"quantity", "boardType", "finish", "card": {"id",
"uniqueCardId", "scryfall_id", "name", "set", "cn", ...}}}}, ...}}.
"""

from typing import Any

from src.data_retrieval.download_utils import pick_keys

# The keys kept at each level, under the API's own names
_KEPT_DECK_KEYS = (
    "publicId", "id", "name", "format", "visibility", "description",
    "createdAtUtc", "lastUpdatedAtUtc", "version", "viewCount", "likeCount",
    "bookmarkCount", "commentCount", "bracket", "autoBracket", "hubs",
    "authorTags", "colorIdentity",
)  # fmt: skip
_KEPT_CARD_ROW_KEYS = ("quantity", "boardType", "finish", "isProxy")
_KEPT_CARD_KEYS = (
    "id", "uniqueCardId", "scryfall_id", "name", "set", "cn", "layout", "lang",
)  # fmt: skip


def prune_deck(payload: Any) -> dict[str, Any]:
    """One deck's JSON pruned, names and nesting unchanged: the deck's
    _KEPT_DECK_KEYS, and "boards" with every board that holds cards, its
    "count" and "cards" ({key: row}), each row's _KEPT_CARD_ROW_KEYS plus
    "card" holding _KEPT_CARD_KEYS. A kept key the payload lacks is left
    out, not invented.

    Inputs: payload (one deck's JSON). Output: the pruned dict.
    Side effects: none.
    Exceptions: KeyError if payload lacks "publicId" or "boards", a board
        lacks "cards", or a row lacks "card"; TypeError for a wrong JSON
        type.

    Example:
        >>> prune_deck({"publicId": "x", "boards": {"mainboard": {"count": 0,
        ...     "cards": {}}}, "createdByUser": {"userName": "y"}})
        {'publicId': 'x', 'boards': {}}
    """
    _require_object(payload, "deck payload")
    payload["publicId"]  # required: KeyError if absent
    boards = _require_object(payload["boards"], "boards")

    result = pick_keys(payload, _KEPT_DECK_KEYS)
    result["boards"] = {
        name: pruned
        for name, board in boards.items()
        if (pruned := _prune_board(board))["cards"]
    }
    return result


def _prune_board(board: Any) -> dict[str, Any]:
    """One board: {"count", "cards": {key: pruned row}} ("count" only if
    the payload has it).
    Inputs: board (one "boards" value). Output: the pruned board.
    Side effects: none. Exceptions: KeyError, TypeError (see prune_deck)."""
    _require_object(board, "board")
    cards = _require_object(board["cards"], "board cards")
    result = pick_keys(board, ("count",))
    result["cards"] = {key: _prune_card_row(entry) for key, entry in cards.items()}
    return result


def _prune_card_row(entry: Any) -> dict[str, Any]:
    """One card row: _KEPT_CARD_ROW_KEYS plus "card" (_KEPT_CARD_KEYS).
    Inputs: entry (one "cards" value). Output: the pruned row.
    Side effects: none.
    Exceptions: KeyError if it lacks "card"; TypeError for a non-object."""
    _require_object(entry, "card row")
    card = _require_object(entry["card"], "card")
    result = pick_keys(entry, _KEPT_CARD_ROW_KEYS)
    result["card"] = pick_keys(card, _KEPT_CARD_KEYS)
    return result


def _require_object(value: Any, what: str) -> dict[str, Any]:
    """value, if it is a JSON object.
    Inputs: value, what (names it in the error). Output: value.
    Side effects: none. Exceptions: TypeError otherwise."""
    if not isinstance(value, dict):
        raise TypeError(f"{what} is {type(value).__name__}, not an object")
    return value
