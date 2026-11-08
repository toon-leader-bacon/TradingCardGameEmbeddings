from collections import Counter
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.deck_sizes import DeckSizes
from src.data_refinement.deck_box.limitless_play.game_spec import ONE_PIECE_SPEC
from src.data_refinement.deck_box.limitless_play.tcg_extraction_stages import (
    ALL_TCG_STAGES,
    LimitlessLorcanaDeckExtractionStage,
    LimitlessOnePieceDeckExtractionStage,
    LimitlessTcgDeckExtractionStage,
)
from src.data_retrieval.limitless_play.downloader import (
    LimitlessGame,
    LimitlessPlayDownloader,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.deck_box.limitless_play.fixtures import (
    raw_entry,
    write_standings,
)
from tests.data_refinement.deck_box.limitless_play.tcg_fixtures import (
    card_uuid_named,
    leader_entry,
    make_tcg_binder,
)

_OP = GameId.ONE_PIECE


def _one_piece_row(character_count: int, archetype: str | None = "Red Luffy") -> dict:
    return {
        "deck": {"id": "x", "name": archetype, "icons": []} if archetype else None,
        "decklist": {
            "leader": leader_entry("Monkey.D.Luffy", "OP13", "001"),
            "character": [raw_entry(character_count, "OP13", "016", "Monkey.D.Garp")],
            "event": [],
            "stage": [],
            "side": [raw_entry(4, "OP01", "099", "Sided Card")],
        },
    }


def _one_piece_binder() -> CardBinder:
    return make_tcg_binder(
        _OP, [("Monkey.D.Luffy", "OP13-001"), ("Monkey.D.Garp", "OP13-016")]
    )


class TestStageClasses:
    def test_constants_come_from_the_spec(self) -> None:
        stage = LimitlessOnePieceDeckExtractionStage
        assert stage.SOURCE_GAME is GameId.ONE_PIECE
        assert stage.DECK_SIZES == DeckSizes(51, 51)
        assert stage.DEFAULT_RAW_PATH == (
            LimitlessPlayDownloader.raw_data_dir_for(LimitlessGame.ONE_PIECE)
            / LimitlessPlayDownloader.STANDINGS_FILENAME
        )
        assert stage.SPEC is ONE_PIECE_SPEC

    def test_a_subclass_without_a_spec_raises(self) -> None:
        with pytest.raises(TypeError, match="SPEC"):

            class _NoSpec(LimitlessTcgDeckExtractionStage):
                pass

    def test_every_stage_is_a_different_game(self) -> None:
        games = [stage.SOURCE_GAME for stage in ALL_TCG_STAGES]
        assert len(games) == len(set(games)) == 8


class TestExtract:
    def test_stores_leader_and_deck_without_the_side_deck(self, tmp_path: Path) -> None:
        binder = _one_piece_binder()
        standings = write_standings(tmp_path, {"t1": [_one_piece_row(50)]})
        box = DeckBox()

        created = LimitlessOnePieceDeckExtractionStage().extract(standings, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "Red Luffy"
        assert Counter(deck.card_nocab_uuids) == {
            card_uuid_named(binder, _OP, "Monkey.D.Luffy"): 1,
            card_uuid_named(binder, _OP, "Monkey.D.Garp"): 50,
        }
        assert deck.provenance is not None
        assert deck.provenance.data_source is DataSource.LIMITLESS_PLAY
        assert deck.provenance.source_id == "t1:0"

    def test_skips_lists_outside_the_games_sizes_and_stores_repeats_once(
        self, tmp_path: Path
    ) -> None:
        binder = _one_piece_binder()
        standings = write_standings(
            tmp_path,
            {
                "t1": [_one_piece_row(49), _one_piece_row(50)],
                "t2": [_one_piece_row(50, archetype=None)],
            },
        )
        box = DeckBox()
        stage = LimitlessOnePieceDeckExtractionStage()

        created = stage.extract(standings, box, binder)

        assert len(created) == 1
        assert list(box.all_uuids(_OP)) == created
        assert stage.extract(standings, box, binder) == []

    def test_a_name_only_game_matches_by_folded_name_and_unknowns_the_rest(
        self, tmp_path: Path
    ) -> None:
        game = GameId.LORCANA
        binder = make_tcg_binder(game, [("Te Ka - Heartless", None)])
        row = {
            "deck": None,
            "decklist": {
                "character": [
                    raw_entry(58, "001", "192", "Te Kā - Heartless"),
                    raw_entry(2, "009", "001", "Not In The Binder"),
                ],
                "action": [],
                "item": [],
            },
        }
        standings = write_standings(tmp_path, {"t1": [row]})
        box = DeckBox()

        created = LimitlessLorcanaDeckExtractionStage().extract(standings, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "t1:0"
        assert Counter(deck.card_nocab_uuids) == {
            card_uuid_named(binder, game, "Te Ka - Heartless"): 58,
            CardBinder.unknown_card_uuid(game): 2,
        }

    def test_raises_without_the_unknown_sentinel(self, tmp_path: Path) -> None:
        standings = write_standings(tmp_path, {"t1": []})
        with pytest.raises(RuntimeError, match="Unknown sentinel"):
            LimitlessOnePieceDeckExtractionStage().extract(
                standings, DeckBox(), CardBinder()
            )
