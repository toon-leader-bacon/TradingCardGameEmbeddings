"""Reads one Deckstats crawl row (src/data_retrieval/deckstats/; public
decks only, private ones live apart in private/) into a CrawledDeck.

RAW SHAPE (the fields read):
    {"deck": {"idsdeck", "name"},
     "entries": [{"zone", "amount", "name", "is_commander",
                  "is_companion", "card": {"is_token"},
                  "printing": {"scryfall_id"}?}, ...]}

WHAT COUNTS AS THE DECK: entries in zones _DECK_ZONES (main deck,
command zone), plus any entry flagged is_commander or is_companion
(Deckstats files companions under "sideboard"). Dropped: sideboard
(between games), maybeboard, tokens, and any card flagged is_token.

CARD IDS: printing.scryfall_id is the Scryfall printing id (absent on
0.2% of entries: those match by name).
"""

from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    DeckSkip,
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
from src.schema.json_types import JsonObject, JsonValue

_DECK_ZONES = frozenset({"main", "commander"})


class DeckstatsDeckParser:
    """A CrawledDeckParser for Deckstats rows. Never skips a whole deck."""

    def parse(self, row: JsonObject) -> CrawledDeck | DeckSkip:
        """One Deckstats row -> its deck.

        Inputs: row (one crawl row, RAW SHAPE above).
        Output: CrawledDeck (only the cards that count as the deck).
        Side effects: none.
        Exceptions: ValueError naming the deck id on a malformed row.

        Example:
            >>> DeckstatsDeckParser().parse(row).source_id
            '1634363'
        """
        deck_id = _deck_id_for_errors(row)
        try:
            deck = require_object(row["deck"])
            entries = tuple(
                parsed
                for entry in require_object_list(row["entries"])
                if _counts_as_deck(entry)
                and (parsed := _parse_entry(entry)) is not None
            )
            return CrawledDeck(
                str(require_int(deck["idsdeck"])),
                optional_str(deck.get("name")),
                entries,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"malformed Deckstats deck {deck_id!r}: {error}"
            ) from error


def _deck_id_for_errors(row: JsonObject) -> JsonValue:
    """row["deck"]["idsdeck"] if readable, else None (for error text).
    Inputs: row. Output: the id or None. Side effects: none.
    Exceptions: none."""
    deck = row.get("deck")
    return deck.get("idsdeck") if isinstance(deck, dict) else None


def _counts_as_deck(entry: JsonObject) -> bool:
    """WHAT COUNTS AS THE DECK (module docstring) for one entry.
    Inputs: entry. Output: bool. Side effects: none. Exceptions: none."""
    card = entry.get("card")
    if isinstance(card, dict) and card.get("is_token") is True:
        return False
    return (
        entry.get("zone") in _DECK_ZONES
        or entry.get("is_commander") is True
        or entry.get("is_companion") is True
    )


def _parse_entry(entry: JsonObject) -> MtgDeckEntry | None:
    """One entry -> its MtgDeckEntry (printing id when present, name), or
    None for an entry holding no copies (amount below 1).
    Inputs: entry. Output: MtgDeckEntry or None. Side effects: none.
    Exceptions: KeyError / TypeError / ValueError on a malformed entry."""
    count = require_int(entry["amount"])
    if count < 1:
        return None
    printing = entry.get("printing")
    printing_id = printing.get("scryfall_id") if isinstance(printing, dict) else None
    return MtgDeckEntry(
        count=count,
        card=MtgCardRef(
            name=require_str(entry["name"]),
            oracle_id=None,
            printing_id=optional_str(printing_id),
        ),
    )
