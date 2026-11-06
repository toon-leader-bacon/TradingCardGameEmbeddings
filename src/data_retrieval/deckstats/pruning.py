"""Unwrapping one Deckstats page (its Inertia page JSON) by pruning:
kept keys keep the site's own names and nesting, and nothing is
interpreted. Owners (name, id, avatar, karma, the legacy URL that holds
the owner id), folders, prices, ads, page config and card text are
dropped.

Page shapes (2026-11-06):
  - "Deck": props {"deck": {"idsdeck", "name", "format_name",
    "visibility", ...}, "sections": [{"section_position", "name",
    "comment"}], "entries": [{"zone", "amount", "name", "idcard",
    "idprintings", "is_commander", ..., "card": {...}, "printing":
    {"idprintings", "set_abbr", "collector_number", "scryfall_id",
    ...}}], "tags": [...]}.
  - "DeckPrivate": props {"deckName", "meta"}; the deck id is in the
    page's "url" (/decks/<id>).
"""

from typing import Any

from src.data_retrieval.deckstats.crawl_plan import deck_id_of_path
from src.data_retrieval.download_utils import pick_keys
from src.data_retrieval.resumable_crawl import FetchOutcome, ItemFound, ItemRejected

PRIVATE_SUBDIRECTORY = "private"

# The keys kept at each level, under the site's own names
_KEPT_DECK_KEYS = (
    "idsdeck", "name", "description", "short_description", "revision",
    "revision_seq", "format_id", "season_id", "format_code", "format_name",
    "format_season", "visibility", "is_public", "added", "updated",
    "number_cards_main", "number_cards_sideboard", "number_cards_companion",
    "likes", "views", "comments", "bracket", "bracket_auto", "is_official",
    "is_highlighted", "colors", "notes",
)  # fmt: skip
# Card properties (game changer, basic land, ...) and repeats of the
# entry's own name and printing id are left to the card binder
_KEPT_ENTRY_KEYS = (
    "zone", "section_position", "card_position", "amount", "name",
    "is_commander", "is_companion", "is_foil", "is_proxy", "comment",
    "idcard", "idprintings",
)  # fmt: skip
_KEPT_CARD_KEYS = ("idcards", "layout", "is_token")
_KEPT_PRINTING_KEYS = ("set_abbr", "collector_number", "scryfall_id")


def classify_page(page: Any) -> FetchOutcome:
    """One page's outcome by its component: "Deck" -> ItemFound(the
    pruned deck); "DeckPrivate" -> ItemFound(build_private_deck_row's row,
    subdirectory PRIVATE_SUBDIRECTORY); any other component ->
    ItemRejected("page <component>").

    Inputs: page (parse_inertia_page's result). Output: FetchOutcome.
    Side effects: none.
    Exceptions: KeyError, TypeError for a page whose shape doesn't fit
        its component (read_outcome rejects it as unreadable).

    Example:
        >>> classify_page({"component": "Error", "props": {}})
        ItemRejected(reason='page Error')
    """
    match page["component"]:
        case "Deck":
            return ItemFound(prune_deck(page["props"]))
        case "DeckPrivate":
            return ItemFound(
                build_private_deck_row(page["props"], page["url"]),
                PRIVATE_SUBDIRECTORY,
            )
        case component:
            return ItemRejected(f"page {component}")


def prune_deck(props: Any) -> dict[str, Any]:
    """A Deck page's props pruned: {"deck": _KEPT_DECK_KEYS, "sections",
    "entries": each _KEPT_ENTRY_KEYS plus "card" (_KEPT_CARD_KEYS) and
    "printing" (_KEPT_PRINTING_KEYS), "tags"}. A kept key the page lacks
    is left out, not invented.

    Inputs: props. Output: the pruned dict. Side effects: none.
    Exceptions: KeyError if props lacks "deck" (or its "idsdeck") or
        "entries"; TypeError for a wrong JSON type.

    Example:
        >>> prune_deck({"deck": {"idsdeck": 1, "owner_name": "x"},
        ...     "entries": [], "ads": {}})
        {'deck': {'idsdeck': 1}, 'entries': []}
    """
    _require_object(props, "deck page props")
    deck = _require_object(props["deck"], "deck")
    if "idsdeck" not in deck:
        raise KeyError("deck has no idsdeck")
    entries = props["entries"]
    if not isinstance(entries, list):
        raise TypeError(f"entries is {type(entries).__name__}, not a list")

    result: dict[str, Any] = {"deck": pick_keys(deck, _KEPT_DECK_KEYS)}
    if isinstance(props.get("sections"), list):
        result["sections"] = props["sections"]
    result["entries"] = [_prune_entry(entry) for entry in entries]
    if isinstance(props.get("tags"), list):
        result["tags"] = props["tags"]
    return result


def build_private_deck_row(props: Any, page_url: Any) -> dict[str, Any]:
    """A DeckPrivate page as {"idsdeck", "deckName", "visibility":
    "private"} (idsdeck an int, as Deck pages carry it).

    Inputs: props (with "deckName"), page_url ("/decks/<id>", or
        "/decks/<id>-<slug>" for a sitemap deck made private since).
    Output: dict. Side effects: none.
    Exceptions: KeyError if props lacks "deckName"; TypeError for a
        wrong type, or a page_url holding no deck id (a malformed page
        is permanent, so it is raised as read_outcome's reject type).

    Example:
        >>> build_private_deck_row({"deckName": "Einkaufsliste"}, "/decks/1705661")
        {'idsdeck': 1705661, 'deckName': 'Einkaufsliste', 'visibility': 'private'}
    """
    _require_object(props, "private page props")
    deck_name = props["deckName"]
    if not isinstance(page_url, str):
        raise TypeError(f"page url is {type(page_url).__name__}, not a string")
    try:
        deck_id = deck_id_of_path(page_url)
    except ValueError as error:
        # A malformed page is a permanent answer: read_outcome rejects a
        # TypeError, while a ValueError would count as transient
        raise TypeError(f"private page url {page_url!r} holds no deck id") from error
    return {"idsdeck": deck_id, "deckName": deck_name, "visibility": "private"}


def _prune_entry(entry: Any) -> dict[str, Any]:
    """One card entry: _KEPT_ENTRY_KEYS, plus "card" and "printing"
    pruned when they are objects.
    Inputs: entry. Output: the pruned entry. Side effects: none.
    Exceptions: TypeError for a non-object entry."""
    _require_object(entry, "card entry")
    result = pick_keys(entry, _KEPT_ENTRY_KEYS)
    for key, kept_keys in (
        ("card", _KEPT_CARD_KEYS),
        ("printing", _KEPT_PRINTING_KEYS),
    ):
        nested = entry.get(key)
        if isinstance(nested, dict):
            result[key] = pick_keys(nested, kept_keys)
    return result


def _require_object(value: Any, what: str) -> dict[str, Any]:
    """value, if it is a JSON object.
    Inputs: value, what (names it in the error). Output: value.
    Side effects: none. Exceptions: TypeError otherwise."""
    if not isinstance(value, dict):
        raise TypeError(f"{what} is {type(value).__name__}, not an object")
    return value
