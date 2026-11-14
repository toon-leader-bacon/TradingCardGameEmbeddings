"""Reads one Moxfield crawl row (src/data_retrieval/moxfield/) into a
CrawledDeck.

RAW SHAPE (the fields read):
    {"publicId", "name", "format",
     "boards": {<board>: {"count", "cards": {<key>: {"quantity",
                "card": {"name", "scryfall_id"}}}}}}

WHAT COUNTS AS THE DECK: the boards in _DECK_BOARDS: the main deck plus
the command zone (commanders, companions, Oathbreaker signature
spells). Dropped: sideboard (between games), maybeboard, tokens,
attractions, stickers, contraptions, schemes, planes.

CARD IDS: card.scryfall_id is the Scryfall printing id.

WHICH DECKS: format "package" lists are card bundles a user shares, not
decks: skipped as NOT_A_DECK.
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
    require_str,
)
from src.schema.json_types import JsonObject

_DECK_BOARDS = ("mainboard", "commanders", "companions", "signatureSpells")
_NOT_A_DECK_FORMATS = frozenset({"package"})


class MoxfieldDeckParser:
    """A CrawledDeckParser for Moxfield rows."""

    def parse(self, row: JsonObject) -> CrawledDeck | DeckSkip:
        """One Moxfield row -> its deck, or why it is skipped.

        Inputs: row (one crawl row, RAW SHAPE above).
        Output: CrawledDeck (only the cards that count as the deck), or
            DeckSkip(NOT_A_DECK).
        Side effects: none.
        Exceptions: ValueError naming the deck id on a malformed row.

        Example:
            >>> MoxfieldDeckParser().parse(row).name
            'Yuna Re-Enchant'
        """
        deck_id = row.get("publicId")
        try:
            if row.get("format") in _NOT_A_DECK_FORMATS:
                return DeckSkip(DeckSkipReason.NOT_A_DECK)
            entries = tuple(
                entry
                for board_name in _DECK_BOARDS
                for board_card in _board_cards(row, board_name)
                if (entry := _parse_entry(board_card)) is not None
            )
            return CrawledDeck(
                require_str(row["publicId"]), optional_str(row.get("name")), entries
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed Moxfield deck {deck_id!r}: {error}") from error


def _board_cards(row: JsonObject, board_name: str) -> list[JsonObject]:
    """The card objects of one board; [] when the deck lacks it.
    Inputs: row, board_name. Output: list. Side effects: none.
    Exceptions: KeyError / TypeError on a malformed board."""
    board = require_object(row["boards"]).get(board_name)
    if board is None:
        return []
    cards = require_object(require_object(board).get("cards") or {})
    return [require_object(board_card) for board_card in cards.values()]


def _parse_entry(board_card: JsonObject) -> MtgDeckEntry | None:
    """One board card -> its entry (printing id and name), or None for a
    card holding no copies (quantity below 1; seen once in 1.7M rows).
    Inputs: board_card. Output: MtgDeckEntry or None. Side effects: none.
    Exceptions: KeyError / TypeError / ValueError on a malformed card."""
    count = require_int(board_card["quantity"])
    if count < 1:
        return None
    card = require_object(board_card["card"])
    return MtgDeckEntry(
        count=count,
        card=MtgCardRef(
            name=require_str(card["name"]),
            oracle_id=None,
            printing_id=optional_str(card.get("scryfall_id")),
        ),
    )
