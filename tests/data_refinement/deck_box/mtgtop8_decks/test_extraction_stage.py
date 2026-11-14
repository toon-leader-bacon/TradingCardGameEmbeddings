from collections import Counter
from pathlib import Path

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.mtgtop8_decks.extraction_stage import (
    MtgTop8DeckExtractionStage,
)
from src.schema.data_source import DataSource
from tests.data_refinement.deck_box.mtg_crawls.fixtures import (
    card_uuid,
    make_mtg_binder,
    write_shards,
)


def _row(deck_id: int, deck_format: str, export: str) -> dict:
    return {
        "deck_id": deck_id,
        "event_id": 1,
        "format": deck_format,
        "deck_name": f"Deck {deck_id}",
        "event_name": "League",
        "rank": "1",
        "date": "09/11/26",
        "mtgo": export,
    }


class TestMtgTop8DeckExtractionStage:
    def test_reads_every_format_directory(self, tmp_path: Path) -> None:
        write_shards(tmp_path / "PAU", [_row(1, "PAU", "60 Mountain\r\n")])
        write_shards(
            tmp_path / "EDH",
            [_row(2, "EDH", "99 Forest\r\nSideboard\r\n1 Slimefoot and Squee\r\n")],
        )
        (tmp_path / "VI").mkdir()  # listed, no shards yet

        rows = list(MtgTop8DeckExtractionStage().shard_rows(tmp_path))

        assert [row["deck_id"] for row in rows] == [2, 1]  # format-code order

    def test_stores_decks_with_the_command_zone(self, tmp_path: Path) -> None:
        binder = make_mtg_binder(
            [
                ("Forest", "o-forest", ()),
                ("Mountain", "o-mountain", ()),
                ("Slimefoot and Squee", "o-slimefoot", ()),
            ]
        )
        write_shards(tmp_path / "PAU", [_row(1, "PAU", "60 Mountain\r\n")])
        write_shards(
            tmp_path / "EDH",
            [_row(2, "EDH", "99 Forest\r\nSideboard\r\n1 Slimefoot and Squee\r\n")],
        )
        box = DeckBox()

        created = MtgTop8DeckExtractionStage().extract(tmp_path, box, binder)

        decks = [box.get_by_uuid(uuid) for uuid in created]
        assert [deck.name for deck in decks if deck is not None] == ["Deck 2", "Deck 1"]
        commander_deck = decks[0]
        assert commander_deck is not None and commander_deck.provenance is not None
        assert commander_deck.provenance.data_source is DataSource.MTGTOP8
        assert commander_deck.provenance.source_id == "2"
        assert Counter(commander_deck.card_nocab_uuids) == {
            card_uuid("o-forest"): 99,
            card_uuid("o-slimefoot"): 1,
        }
