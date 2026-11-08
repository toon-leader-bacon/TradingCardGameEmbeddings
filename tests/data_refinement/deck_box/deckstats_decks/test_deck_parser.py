import pytest

from src.data_refinement.deck_box.deckstats_decks.deck_parser import (
    DeckstatsDeckParser,
)
from src.data_refinement.deck_box.deckstats_decks.extraction_stage import (
    DeckstatsDeckExtractionStage,
)
from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    MtgCardRef,
    MtgDeckEntry,
)
from src.schema.data_source import DataSource


def _entry(zone: str, name: str, amount: int = 1, **extra: object) -> dict:
    return {
        "zone": zone,
        "amount": amount,
        "name": name,
        "is_commander": False,
        "is_companion": False,
        "card": {"idcards": 1, "layout": "normal", "is_token": False},
        "printing": {"set_abbr": "S", "scryfall_id": f"p-{name}"},
        **extra,
    }


def _deck(entries: list[dict]) -> dict:
    return {
        "deck": {"idsdeck": 1634363, "name": "K'rrik"},
        "sections": [],
        "entries": entries,
    }


class TestDeckstatsDeckParser:
    def test_keeps_main_command_zone_and_flagged_cards(self) -> None:
        entries = [
            _entry("main", "Swamp", amount=30),
            _entry("commander", "K'rrik"),
            _entry("sideboard", "Lurrus", is_companion=True),
            _entry("sideboard", "Partner", is_commander=True),
            _entry("sideboard", "Sided"),
            _entry("maybeboard", "Maybe"),
            _entry("tokens", "Zombie"),
            _entry("main", "Token In Main", card={"is_token": True}),
        ]

        deck = DeckstatsDeckParser().parse(_deck(entries))

        assert isinstance(deck, CrawledDeck)
        assert (deck.source_id, deck.name) == ("1634363", "K'rrik")
        assert [entry.card.name for entry in deck.entries] == [
            "Swamp", "K'rrik", "Lurrus", "Partner",
        ]  # fmt: skip
        assert deck.entries[0] == MtgDeckEntry(30, MtgCardRef("Swamp", None, "p-Swamp"))

    def test_an_entry_without_a_printing_matches_by_name(self) -> None:
        deck = DeckstatsDeckParser().parse(
            _deck([_entry("main", "Odd", printing=None)])
        )

        assert isinstance(deck, CrawledDeck)
        assert deck.entries[0].card == MtgCardRef("Odd", None, None)

    def test_a_malformed_row_names_its_deck(self) -> None:
        with pytest.raises(ValueError, match="1634363"):
            row = _deck([_entry("main", "x", amount="1")])  # type: ignore[arg-type]
            DeckstatsDeckParser().parse(row)


class TestDeckstatsDeckExtractionStage:
    def test_wiring(self) -> None:
        assert DeckstatsDeckExtractionStage.DATA_SOURCE is DataSource.DECKSTATS


class TestDeckstatsCopyCounts:
    def test_an_entry_with_no_copies_is_dropped(self) -> None:
        entries = [_entry("main", "Ghost", amount=0), _entry("main", "Bolt")]

        deck = DeckstatsDeckParser().parse(_deck(entries))

        assert isinstance(deck, CrawledDeck)
        assert [entry.card.name for entry in deck.entries] == ["Bolt"]
