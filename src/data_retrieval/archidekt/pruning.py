"""Unwrapping one Archidekt deck's JSON (~280 KB for a 100-card deck:
prices, card text, combo data) by pruning: the kept keys keep the API's
own names and nesting, and nothing is interpreted.

What is dropped: bulk (prices, card text, combo data, images), the
owner, what the card binder already holds per card (display name,
release date, edition name), per-row bookkeeping (Archidekt's row id,
the row's createdAt), the Scryfall printing id (card.uid: the edition
code and collector number still name the printing), and null values
(an absent key reads as null). The oracle id (card.oracleCard.uid), the
cards are matched on, is kept. Measured on 2,000 real rows: ~42 KB raw,
~8.5 KB gzipped per deck, down from ~70 KB and ~13.5 KB.

Rows landed before this trim (the first 17,520 sitemap decks) carry the
dropped keys too; prune_deck() on such a row gives this shape.
"""

from typing import Any

from src.data_retrieval.download_utils import pick_keys

# The keys kept at each level, under the API's own names
_KEPT_DECK_KEYS = (
    "id", "name", "createdAt", "updatedAt", "deckFormat", "edhBracket",
    "game", "description", "viewCount", "hasPrimer", "private",
    "unlisted", "theorycrafted", "categories", "deckTags", "customCards",
)  # fmt: skip
_KEPT_CARD_ROW_KEYS = (
    "quantity", "modifier", "categories", "removedCategories", "companion",
    "label", "notes", "customCmc", "deletedAt",
)  # fmt: skip
_KEPT_CARD_KEYS = ("collectorNumber",)
_KEPT_EDITION_KEYS = ("editioncode",)
_KEPT_ORACLE_CARD_KEYS = ("uid", "name", "layout")


def prune_deck(payload: Any) -> dict[str, Any]:
    """One deck's JSON pruned to the kept keys, names and nesting
    unchanged: the deck's _KEPT_DECK_KEYS, and "cards" with each row's
    _KEPT_CARD_ROW_KEYS plus "card" holding _KEPT_CARD_KEYS,
    "edition" (_KEPT_EDITION_KEYS) and "oracleCard"
    (_KEPT_ORACLE_CARD_KEYS). A kept key the payload lacks, or holds as
    null, is left out, not invented. Values are copied, never
    interpreted.

    Inputs: payload (one deck's JSON). Output: the pruned dict.
    Side effects: none.
    Exceptions: KeyError if payload lacks "id" or "cards", or a card row
        lacks "card"; TypeError if they have the wrong JSON type.

    Example:
        >>> prune_deck({"id": 1, "cards": [], "owner": {"id": 9}, "game": None})
        {'id': 1, 'cards': []}
    """
    if not isinstance(payload, dict):
        raise TypeError(f"deck payload is {type(payload).__name__}, not an object")
    if "id" not in payload:
        raise KeyError("deck has no id")
    raw_cards = payload["cards"]
    if not isinstance(raw_cards, list):
        raise TypeError(f"deck cards is {type(raw_cards).__name__}, not a list")

    result = _pick_present(payload, _KEPT_DECK_KEYS)
    result["cards"] = [_prune_card_row(entry) for entry in raw_cards]
    return result


def _prune_card_row(entry: Any) -> dict[str, Any]:
    """One "cards" row pruned (see prune_deck).
    Inputs: entry. Output: the pruned row. Side effects: none.
    Exceptions: KeyError if it lacks "card"; TypeError for a non-object
        row or card."""
    if not isinstance(entry, dict):
        raise TypeError(f"card row is {type(entry).__name__}, not an object")
    card = entry["card"]
    if not isinstance(card, dict):
        raise TypeError(f"card is {type(card).__name__}, not an object")

    result = _pick_present(entry, _KEPT_CARD_ROW_KEYS)
    kept_card = _pick_present(card, _KEPT_CARD_KEYS)
    for key, kept_keys in (
        ("edition", _KEPT_EDITION_KEYS),
        ("oracleCard", _KEPT_ORACLE_CARD_KEYS),
    ):
        nested = card.get(key)
        if isinstance(nested, dict):
            kept_card[key] = _pick_present(nested, kept_keys)
    result["card"] = kept_card
    return result


def _pick_present(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """pick_keys without the null values.
    Inputs: source, keys. Output: a new dict. Side effects: none.
    Exceptions: none."""
    return {
        key: value
        for key, value in pick_keys(source, keys).items()
        if value is not None
    }
