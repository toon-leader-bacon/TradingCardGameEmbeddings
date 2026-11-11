from typing import Any

import pytest

from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    MtgCardRef,
    MtgDeckEntry,
)
from src.data_refinement.deck_box.mtgtop8_decks.deck_parser import MtgTop8DeckParser


def _row(export: str, deck_format: str = "PAU", **extra: Any) -> dict[str, Any]:
    return {
        "deck_id": 909096,
        "event_id": 93328,
        "format": deck_format,
        "deck_name": "Red Deck Wins",
        "event_name": "League",
        "rank": "1",
        "date": "09/11/26",
        "mtgo": export,
        **extra,
    }


def _entry(count: int, name: str) -> MtgDeckEntry:
    return MtgDeckEntry(count, MtgCardRef(name, None, None))


def _parse(export: str, deck_format: str = "PAU", **extra: Any) -> CrawledDeck:
    parsed = MtgTop8DeckParser().parse(_row(export, deck_format, **extra))
    assert isinstance(parsed, CrawledDeck)
    return parsed


_CONSTRUCTED = (
    "4 Lightning Bolt\r\n56 Mountain\r\n"
    "Sideboard\r\n2 Lurrus of the Dream-Den\r\n3 Pyroblast\r\n"
)


class TestParse:
    def test_constructed_keeps_main_and_one_companion(self) -> None:
        deck = _parse(_CONSTRUCTED)

        assert deck == CrawledDeck(
            "909096",
            "Red Deck Wins",
            (
                _entry(4, "Lightning Bolt"),
                _entry(56, "Mountain"),
                _entry(1, "Lurrus of the Dream-Den"),
            ),
        )

    @pytest.mark.parametrize("deck_format", ["EDH", "cEDH", "EDHM", "EDHP"])
    def test_commander_formats_keep_the_command_zone(self, deck_format: str) -> None:
        export = "98 Forest\r\nSideboard\r\n1 Kraum, Ludevic's Opus\r\n1 Tymna the Weaver\r\n"

        deck = _parse(export, deck_format)

        assert deck.card_count == 100
        assert deck.entries[-2:] == (
            _entry(1, "Kraum, Ludevic's Opus"),
            _entry(1, "Tymna the Weaver"),
        )

    def test_limited_keeps_no_sideboard_card(self) -> None:
        deck = _parse(_CONSTRUCTED, "LI")

        assert deck.entries == (_entry(4, "Lightning Bolt"), _entry(56, "Mountain"))

    def test_split_cards_use_the_binder_spelling(self) -> None:
        deck = _parse("4 Fire/Ice\r\n2 Wear // Tear\r\n1 Who/What/When/Where/Why\r\n")

        assert [entry.card.name for entry in deck.entries] == [
            "Fire // Ice",
            "Wear // Tear",
            "Who // What // When // Where // Why",
        ]

    def test_blank_lines_and_empty_counts_are_dropped(self) -> None:
        deck = _parse("\r\n4 Lightning Bolt\r\n0 Mountain\r\n\r\nSideboard\r\n")

        assert deck.entries == (_entry(4, "Lightning Bolt"),)

    @pytest.mark.parametrize("name", [None, ""])
    def test_an_unnamed_deck_has_no_name(self, name: str | None) -> None:
        assert _parse(_CONSTRUCTED, deck_name=name).name is None

    @pytest.mark.parametrize(
        "row",
        [
            _row("4 Lightning Bolt\r\nBolt\r\n"),
            _row("Sideboard\r\n1 A\r\nSideboard\r\n"),
            _row(_CONSTRUCTED, "XYZ"),
            _row(_CONSTRUCTED, deck_id="909096"),
            {key: value for key, value in _row(_CONSTRUCTED).items() if key != "mtgo"},
        ],
    )
    def test_a_malformed_row_raises_naming_the_deck(self, row: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="malformed MTGTop8 deck"):
            MtgTop8DeckParser().parse(row)
