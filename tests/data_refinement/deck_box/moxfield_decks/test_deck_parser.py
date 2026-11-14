import pytest

from src.data_refinement.deck_box.moxfield_decks.deck_parser import MoxfieldDeckParser
from src.data_refinement.deck_box.moxfield_decks.extraction_stage import (
    MoxfieldDeckExtractionStage,
)
from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    DeckSkip,
    DeckSkipReason,
    MtgCardRef,
    MtgDeckEntry,
)
from src.schema.data_source import DataSource


def _board(*cards: tuple[int, str]) -> dict:
    return {
        "count": sum(quantity for quantity, _ in cards),
        "cards": {
            f"k{name}": {
                "quantity": quantity,
                "boardType": "x",
                "card": {"name": name, "scryfall_id": f"p-{name}", "set": "s"},
            }
            for quantity, name in cards
        },
    }


def _deck(boards: dict, **extra: object) -> dict:
    return {
        "publicId": "Ayk",
        "name": "Yuna",
        "format": "commander",
        "boards": boards,
        **extra,
    }


class TestMoxfieldDeckParser:
    def test_reads_the_deck_and_command_zone_boards_only(self) -> None:
        boards = {
            "mainboard": _board((9, "Plains")),
            "commanders": _board((1, "Yuna")),
            "companions": _board((1, "Lurrus")),
            "signatureSpells": _board((1, "Spell")),
            "sideboard": _board((2, "Sided")),
            "maybeboard": _board((1, "Maybe")),
            "tokens": _board((1, "Soldier")),
            "attractions": _board((1, "Attraction")),
        }

        deck = MoxfieldDeckParser().parse(_deck(boards))

        assert isinstance(deck, CrawledDeck)
        assert (deck.source_id, deck.name, deck.card_count) == ("Ayk", "Yuna", 12)
        assert [entry.card.name for entry in deck.entries] == [
            "Plains", "Yuna", "Lurrus", "Spell",
        ]  # fmt: skip
        assert deck.entries[0] == MtgDeckEntry(
            9, MtgCardRef("Plains", None, "p-Plains")
        )

    def test_package_lists_are_not_decks(self) -> None:
        row = _deck({"mainboard": _board((1, "x"))}, format="package")
        assert MoxfieldDeckParser().parse(row) == DeckSkip(DeckSkipReason.NOT_A_DECK)

    def test_a_malformed_row_names_its_deck(self) -> None:
        with pytest.raises(ValueError, match="Ayk"):
            MoxfieldDeckParser().parse(_deck({"mainboard": {"cards": [1]}}))


class TestMoxfieldDeckExtractionStage:
    def test_wiring(self) -> None:
        assert MoxfieldDeckExtractionStage.DATA_SOURCE is DataSource.MOXFIELD


class TestMoxfieldCopyCounts:
    def test_a_card_with_no_copies_is_dropped(self) -> None:
        boards = {"mainboard": _board((0, "Ghost"), (4, "Bolt"))}

        deck = MoxfieldDeckParser().parse(_deck(boards))

        assert isinstance(deck, CrawledDeck)
        assert [entry.card.name for entry in deck.entries] == ["Bolt"]

    def test_an_unnamed_deck_has_no_name(self) -> None:
        deck = MoxfieldDeckParser().parse(_deck({}, name=None))
        assert isinstance(deck, CrawledDeck) and deck.name is None
