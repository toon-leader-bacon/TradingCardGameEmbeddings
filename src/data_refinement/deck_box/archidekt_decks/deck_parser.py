"""Reads one Archidekt crawl row (src/data_retrieval/archidekt/, pruned
by its pruning.py) into a CrawledDeck.

RAW SHAPE (the fields read):
    {"id", "name", "unlisted",
     "categories": [{"name", "includedInDeck"}, ...],
     "cards": [{"quantity", "categories": [name, ...], "companion",
                "deletedAt"?,
                "card": {"oracleCard": {"uid", "name"}}}, ...]}

WHAT COUNTS AS THE DECK: every card row except one whose primary
(first) category the deck lists with includedInDeck false (Archidekt's
Maybeboard, and most users' Sideboard). Uncategorized rows and rows
whose category the deck doesn't list count, as Archidekt counts them. A
row flagged companion always counts (companions are often filed under
Sideboard). A row with a deletedAt is a card removed from the deck and
is dropped.

CARD IDS: oracleCard.uid is the Scryfall oracle id; the printing id is
not kept by the crawl (pruning.py).

WHICH DECKS: unlisted decks are skipped unless include_unlisted (their
authors chose not to list them; kept out of published training data).
"""

from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    DeckSkip,
    DeckSkipReason,
    MtgCardRef,
    MtgDeckEntry,
)
from src.schema.json_checks import (
    optional_str,
    require_int,
    require_object,
    require_object_list,
    require_str,
)
from src.schema.json_types import JsonObject


class ArchidektDeckParser:
    """A CrawledDeckParser for Archidekt rows."""

    def __init__(self, include_unlisted: bool = False) -> None:
        """
        Inputs: include_unlisted (keep decks their authors left
            unlisted; default False).
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self._include_unlisted = include_unlisted

    def parse(self, row: JsonObject) -> CrawledDeck | DeckSkip:
        """One Archidekt row -> its deck, or why it is skipped.

        Inputs: row (one crawl row, RAW SHAPE above).
        Output: CrawledDeck (only the cards that count as the deck), or
            DeckSkip(UNLISTED).
        Side effects: none.
        Exceptions: ValueError naming the deck id on a malformed row.

        Example:
            >>> ArchidektDeckParser().parse(row).name
            'Baby Lasagna'
        """
        deck_id = row.get("id")
        try:
            if row.get("unlisted") and not self._include_unlisted:
                return DeckSkip(DeckSkipReason.UNLISTED)
            excluded_categories = _excluded_category_names(row)
            entries = tuple(
                entry
                for card_row in require_object_list(row["cards"])
                if _counts_as_deck(card_row, excluded_categories)
                and (entry := _parse_entry(card_row)) is not None
            )
            return CrawledDeck(
                str(require_int(row["id"])), optional_str(row.get("name")), entries
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"malformed Archidekt deck {deck_id!r}: {error}"
            ) from error


def _excluded_category_names(row: JsonObject) -> frozenset[str]:
    """The deck's category names whose includedInDeck is false.
    Inputs: row. Output: frozenset of names. Side effects: none.
    Exceptions: KeyError / TypeError on a malformed categories list."""
    return frozenset(
        require_str(category["name"])
        for category in require_object_list(row.get("categories") or [])
        if category.get("includedInDeck") is False
    )


def _counts_as_deck(card_row: JsonObject, excluded_categories: frozenset[str]) -> bool:
    """WHAT COUNTS AS THE DECK (module docstring) for one card row.
    Inputs: card_row, excluded_categories. Output: bool.
    Side effects: none. Exceptions: TypeError on malformed categories."""
    if card_row.get("deletedAt"):
        return False
    if card_row.get("companion") is True:
        return True
    categories = card_row.get("categories") or []
    if not isinstance(categories, list):
        raise TypeError(f"categories is not a list: {categories!r}")
    if not categories:
        return True
    return require_str(categories[0]) not in excluded_categories


def _parse_entry(card_row: JsonObject) -> MtgDeckEntry | None:
    """One card row -> its entry (oracle id and name; no printing id), or
    None for a row holding no copies (quantity below 1).
    Inputs: card_row. Output: MtgDeckEntry or None. Side effects: none.
    Exceptions: KeyError / TypeError / ValueError on a malformed row."""
    count = require_int(card_row["quantity"])
    if count < 1:
        return None
    oracle_card = require_object(require_object(card_row["card"])["oracleCard"])
    return MtgDeckEntry(
        count=count,
        card=MtgCardRef(
            name=require_str(oracle_card["name"]),
            oracle_id=require_str(oracle_card["uid"]),
            printing_id=None,
        ),
    )
