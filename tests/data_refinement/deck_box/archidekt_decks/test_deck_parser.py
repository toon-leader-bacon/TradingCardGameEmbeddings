import pytest

from src.data_refinement.deck_box.archidekt_decks.deck_parser import (
    ArchidektDeckParser,
)
from src.data_refinement.deck_box.archidekt_decks.extraction_stage import (
    ArchidektDeckExtractionStage,
)
from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    DeckSkip,
    DeckSkipReason,
    MtgCardRef,
    MtgDeckEntry,
)
from src.schema.data_source import DataSource


def _card(quantity: int, name: str, categories: list[str], **extra: object) -> dict:
    return {
        "quantity": quantity,
        "categories": categories,
        "card": {
            "collectorNumber": "1",
            "oracleCard": {"uid": f"o-{name}", "name": name, "layout": "normal"},
        },
        **extra,
    }


def _deck(cards: list[dict], **extra: object) -> dict:
    return {
        "id": 1585124,
        "name": "Baby Lasagna",
        "unlisted": False,
        "categories": [
            {"name": "Ramp", "includedInDeck": True},
            {"name": "Commander", "includedInDeck": True},
            {"name": "Maybeboard", "includedInDeck": False},
            {"name": "Sideboard", "includedInDeck": False},
        ],
        "cards": cards,
        **extra,
    }


def _names(deck: CrawledDeck | DeckSkip) -> list[str]:
    assert isinstance(deck, CrawledDeck)
    return [entry.card.name for entry in deck.entries]


class TestArchidektDeckParser:
    def test_reads_a_deck_with_oracle_ids(self) -> None:
        deck = ArchidektDeckParser().parse(_deck([_card(4, "Bolt", ["Ramp"])]))

        assert deck == CrawledDeck(
            "1585124",
            "Baby Lasagna",
            (MtgDeckEntry(4, MtgCardRef("Bolt", "o-Bolt", None)),),
        )

    def test_keeps_only_the_cards_in_the_deck(self) -> None:
        cards = [
            _card(1, "Commander", ["Commander"]),
            _card(1, "Uncategorized", []),
            _card(1, "Unlisted Category", ["Combo Pieces"]),
            _card(1, "Secondary Maybe", ["Ramp", "Maybeboard"]),
            _card(1, "Maybe", ["Maybeboard"]),
            _card(1, "Sided", ["Sideboard"]),
            _card(1, "Companion", ["Sideboard"], companion=True),
            _card(1, "Removed", ["Ramp"], deletedAt="2024-01-01"),
        ]

        names = _names(ArchidektDeckParser().parse(_deck(cards)))

        assert names == [
            "Commander", "Uncategorized", "Unlisted Category",
            "Secondary Maybe", "Companion",
        ]  # fmt: skip

    def test_unlisted_decks_are_skipped_unless_asked_for(self) -> None:
        row = _deck([_card(1, "Bolt", [])], unlisted=True)

        assert ArchidektDeckParser().parse(row) == DeckSkip(DeckSkipReason.UNLISTED)
        assert _names(ArchidektDeckParser(include_unlisted=True).parse(row)) == ["Bolt"]

    def test_a_malformed_row_names_its_deck(self) -> None:
        row = _deck([{"quantity": "4", "categories": [], "card": {}}])
        with pytest.raises(ValueError, match="1585124"):
            ArchidektDeckParser().parse(row)


class TestArchidektDeckExtractionStage:
    def test_wiring(self) -> None:
        stage = ArchidektDeckExtractionStage()
        assert stage.DATA_SOURCE is DataSource.ARCHIDEKT
        assert str(stage.DEFAULT_RAW_PATH).replace("\\", "/") == "data/raw/archidekt"


class TestArchidektCopyCounts:
    def test_a_row_with_no_copies_is_dropped(self) -> None:
        cards = [_card(0, "Ghost", []), _card(1, "Bolt", [])]
        assert _names(ArchidektDeckParser().parse(_deck(cards))) == ["Bolt"]
