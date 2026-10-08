import gzip
import json
from pathlib import Path

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.data_refinement.deck_box.sts2runs.extraction_stage import (
    Sts2RunsDeckExtractionStage,
)
from src.data_refinement.metrics.sts2_runs.pick_choice import PickKind
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.run_record import RunOutcome
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.metrics.sts2_runs._runs import binder, raw_run


@pytest.fixture
def parser(tmp_path: Path) -> Sts2RunParser:
    return Sts2RunParser(binder(tmp_path), SpireCodexRunsDeckExtractionStage())


class TestPickChoices:
    def test_each_kind_is_read_by_its_own_reader(self, parser: Sts2RunParser) -> None:
        stock = [f"CARD.S{i}" for i in range(1, 7)]
        shop = {
            "rooms": [{"room_type": "shop"}],
            "player_stats": [
                {
                    "player_id": 1,
                    "card_choices": [
                        {"card": {"id": c}, "was_picked": False} for c in stock
                    ],
                    "cards_gained": [{"id": "CARD.S7"}],
                    "cards_removed": [
                        {"id": "CARD.STRIKE_SILENT", "floor_added_to_deck": 1}
                    ],
                }
            ],
        }
        rest = {
            "rooms": [{"room_type": "rest_site"}],
            "player_stats": [
                {
                    "player_id": 1,
                    "rest_site_choices": ["SMITH"],
                    "upgraded_cards": ["CARD.DEFEND_SILENT"],
                }
            ],
        }
        raw = raw_run()
        raw["map_point_history"][1] += [shop, rest]

        choices = parser.parse(raw).players[0].pick_choices

        assert {kind: len(found) for kind, found in choices.items()} == {
            PickKind.CARD_REWARD: 2,
            PickKind.SHOP_PURCHASE: 1,
            PickKind.CARD_REMOVAL: 1,
            PickKind.CARD_UPGRADE: 1,
        }
        assert [c.floor for c in choices[PickKind.CARD_REMOVAL]] == [5]
        assert [c.floor for c in choices[PickKind.CARD_UPGRADE]] == [6]


class TestOutcome:
    def test_a_loss_names_its_killer_encounter(self, parser: Sts2RunParser) -> None:
        run = parser.parse(raw_run())
        assert run.outcome is RunOutcome.LOSS
        assert run.killed_by == "ENCOUNTER.THE_KIN_BOSS"
        assert not run.win

    def test_a_loss_falls_back_to_the_killer_event(self, parser: Sts2RunParser) -> None:
        run = parser.parse(
            raw_run(killed_by_encounter="NONE.NONE", killed_by_event="EVENT.PUNCH_OFF")
        )
        assert run.killed_by == "EVENT.PUNCH_OFF"

    def test_a_win_has_no_killer(self, parser: Sts2RunParser) -> None:
        run = parser.parse(raw_run(win=True, killed_by_encounter="NONE.NONE"))
        assert run.outcome is RunOutcome.WIN
        assert run.killed_by is None

    def test_an_abandoned_run_is_neither_win_nor_loss(
        self, parser: Sts2RunParser
    ) -> None:
        run = parser.parse(raw_run(was_abandoned=True))
        assert run.outcome is RunOutcome.ABANDONED
        assert run.killed_by is None


class TestDerivedStats:
    def test_run_level_stats_come_from_the_map_point_history(
        self, parser: Sts2RunParser
    ) -> None:
        run = parser.parse(raw_run())
        assert run.floors_per_act == (2, 2)
        assert run.total_turns == 14
        assert run.total_combats == 4
        # Two elites entered; the run died in the second
        assert run.elites_killed == 1
        assert run.floors_cleared == 3
        assert run.act_2_start_floor == 3

    def test_a_won_run_clears_every_floor_and_elite(
        self, parser: Sts2RunParser
    ) -> None:
        run = parser.parse(raw_run(win=True))
        assert run.floors_cleared == 4
        assert run.elites_killed == 2

    def test_a_one_act_run_never_reached_act_2(self, parser: Sts2RunParser) -> None:
        run = parser.parse(
            raw_run(map_point_history=[raw_run()["map_point_history"][0]])
        )
        assert run.act_2_start_floor is None

    def test_player_stats_sum_the_players_own_map_point_entries(
        self, parser: Sts2RunParser
    ) -> None:
        (player,) = parser.parse(raw_run()).players
        assert player.damage_taken == 52
        assert player.cards_picked == 2
        assert player.cards_skipped == 2
        assert player.relic_count == 2
        assert player.character == "CHARACTER.SILENT"


class TestDeck:
    def test_deck_uuid_is_the_one_the_deck_box_stage_stored(
        self, parser: Sts2RunParser
    ) -> None:
        (player,) = parser.parse(raw_run()).players
        assert player.deck_uuid == SpireCodexRunsDeckExtractionStage().deck_uuid(
            "abc123", 0
        )

    def test_sts2runs_runs_use_their_own_id_key_and_namespace(
        self, tmp_path: Path
    ) -> None:
        stage = Sts2RunsDeckExtractionStage()
        parser = Sts2RunParser(binder(tmp_path), stage)
        run = parser.parse(raw_run(_serverId=42))
        assert run.run_id == "42"
        assert run.players[0].deck_uuid == stage.deck_uuid("42", 0)

    def test_slots_carry_card_floor_and_upgrade(
        self, parser: Sts2RunParser, tmp_path: Path
    ) -> None:
        lookup = binder(tmp_path)
        strike = lookup.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "STRIKE_SILENT"
        )
        assert strike is not None
        deck = parser.parse(raw_run()).players[0].deck
        assert deck[0].card_uuid == strike.nocab_uuid
        assert [slot.upgraded for slot in deck] == [False, True, False, False]
        assert [slot.floor_added for slot in deck] == [1, 1, 3, 2]

    def test_an_unaliased_card_has_no_card_uuid(self, parser: Sts2RunParser) -> None:
        player = parser.parse(raw_run()).players[0]
        assert player.deck[3].card_uuid is None
        assert len(player.known_card_uuids()) == 3

    def test_a_missing_deck_field_raises(self, parser: Sts2RunParser) -> None:
        run = raw_run()
        del run["players"][0]["deck"][0]["floor_added_to_deck"]
        with pytest.raises(KeyError):
            parser.parse(run)


def test_parsed_deck_uuids_are_the_decks_extract_stores(tmp_path: Path) -> None:
    # The deck-label metrics' rows point at these decks in the published box
    lookup = binder(tmp_path)
    lookup.ensure_unknown_card(GameId.SLAY_THE_SPIRE_2)
    stage = Sts2RunsDeckExtractionStage()
    row = raw_run(_serverId=42)
    raw_path = tmp_path / "runs.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as raw_file:
        raw_file.write(json.dumps(row) + "\n")
    box = DeckBox()
    stage.extract(raw_path, box, lookup)

    (player,) = Sts2RunParser(lookup, stage).parse(row).players

    assert box.get_by_uuid(player.deck_uuid) is not None
