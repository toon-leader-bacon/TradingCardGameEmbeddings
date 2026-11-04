from collections import Counter
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.limitless_play.extraction_stage import (
    LimitlessPlayDeckExtractionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.deck_box.limitless_play.fixtures import (
    card_uuid,
    make_binder,
    raw_entry,
    standings_row,
    write_sets,
    write_standings,
)


class TestExtract:
    def test_stores_a_full_deck_as_a_copy_count_multiset(self, tmp_path: Path) -> None:
        binder = make_binder([("sv4-1", "A"), ("sv1-2", "B")])
        standings = write_standings(
            tmp_path,
            {
                "t1": [
                    standings_row(
                        [raw_entry(56, "PAR", "1", "A"), raw_entry(4, "SVI", "2", "B")]
                    )
                ]
            },
        )
        box = DeckBox()

        created = LimitlessPlayDeckExtractionStage(write_sets(tmp_path)).extract(
            standings, box, binder
        )

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "Charizard"
        assert Counter(deck.card_nocab_uuids) == {
            card_uuid(binder, "sv4-1"): 56,
            card_uuid(binder, "sv1-2"): 4,
        }
        assert deck.provenance is not None
        assert deck.provenance.data_source == DataSource.LIMITLESS_PLAY
        assert deck.provenance.source_id == "t1:0"

    def test_identical_lists_are_stored_once(self, tmp_path: Path) -> None:
        binder = make_binder([("sv4-1", "A")])
        same = standings_row([raw_entry(60, "PAR", "1", "A")])
        standings = write_standings(tmp_path, {"t1": [same, same], "t2": [same]})
        box = DeckBox()

        created = LimitlessPlayDeckExtractionStage(write_sets(tmp_path)).extract(
            standings, box, binder
        )

        assert len(created) == 1
        assert list(box.all_uuids(GameId.POKEMON)) == created

    def test_rerun_creates_nothing(self, tmp_path: Path) -> None:
        binder = make_binder([("sv4-1", "A")])
        standings = write_standings(
            tmp_path, {"t1": [standings_row([raw_entry(60, "PAR", "1", "A")])]}
        )
        stage = LimitlessPlayDeckExtractionStage(write_sets(tmp_path))
        box = DeckBox()
        stage.extract(standings, box, binder)

        assert stage.extract(standings, box, binder) == []

    def test_skips_decklists_not_totalling_60(self, tmp_path: Path) -> None:
        binder = make_binder([("sv4-1", "A")])
        standings = write_standings(
            tmp_path,
            {
                "t1": [
                    standings_row([raw_entry(59, "PAR", "1", "A")]),
                    {"decklist": None, "deck": None},
                ]
            },
        )
        box = DeckBox()

        created = LimitlessPlayDeckExtractionStage(write_sets(tmp_path)).extract(
            standings, box, binder
        )

        assert created == []

    def test_unmatched_cards_become_the_unknown_sentinel(self, tmp_path: Path) -> None:
        binder = make_binder([("sv4-1", "A")])
        standings = write_standings(
            tmp_path,
            {
                "t1": [
                    standings_row(
                        [
                            raw_entry(58, "PAR", "1", "A"),
                            raw_entry(2, "ZZZ", "9", "Nope"),
                        ],
                        archetype=None,
                    )
                ]
            },
        )
        box = DeckBox()

        created = LimitlessPlayDeckExtractionStage(write_sets(tmp_path)).extract(
            standings, box, binder
        )

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "t1:0"
        unknown_uuid = CardBinder.unknown_card_uuid(GameId.POKEMON)
        assert Counter(deck.card_nocab_uuids)[unknown_uuid] == 2

    def test_raises_without_the_unknown_sentinel(self, tmp_path: Path) -> None:
        standings = write_standings(tmp_path, {"t1": []})

        with pytest.raises(RuntimeError, match="Unknown sentinel"):
            LimitlessPlayDeckExtractionStage(write_sets(tmp_path)).extract(
                standings, DeckBox(), CardBinder()
            )

    def test_raises_on_missing_standings_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LimitlessPlayDeckExtractionStage(write_sets(tmp_path)).extract(
                tmp_path / "missing.jsonl", DeckBox(), make_binder([])
            )
