"""Tests for replay_turn_event_rate_metrics.py's
CombatKillInvolvementRateMetric/CombatDamagePushThroughRateMetric - the
base ReplayTurnEventRateMetric's shared accumulate()/finalize() shape
is exercised through these concretes, mirroring
tests/data_refinement/metrics/seventeenlands/game_data/test_game_card_average_metrics.py's
convention of testing a Template Method base only through its
subclasses.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metrics import (
    CombatDamagePushThroughRateMetric,
    CombatKillInvolvementRateMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = [
    "user_turn_1_creatures_attacked",
    "user_turn_1_creatures_blocking",
    "user_turn_1_creatures_unblocked",
    "user_turn_1_user_creatures_killed_combat",
    "user_turn_1_oppo_creatures_killed_combat",
    "user_turn_2_creatures_attacked",
    "user_turn_2_creatures_blocking",
    "user_turn_2_creatures_unblocked",
    "user_turn_2_user_creatures_killed_combat",
    "user_turn_2_oppo_creatures_killed_combat",
    "oppo_turn_1_creatures_attacked",
    "oppo_turn_1_creatures_blocking",
    "oppo_turn_1_creatures_unblocked",
    "oppo_turn_1_user_creatures_killed_combat",
    "oppo_turn_1_oppo_creatures_killed_combat",
]

_NA_ROW: dict = {column: float("nan") for column in _HEADER}


def _make_card(name: str, arena_id: str) -> tuple[GenericCard, str]:
    return (
        GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.MTG,
            name=name,
            raw_content={},
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id=name,
                fetched_at=datetime.now(timezone.utc),
            ),
        ),
        arena_id,
    )


def _binder_with_cards(names_and_ids: list[tuple[str, str]]) -> CardBinder:
    binder = CardBinder()
    for name, arena_id in names_and_ids:
        card, _ = _make_card(name, arena_id)
        binder.create(card)
        binder.register_alias(GameId.MTG, DataSource.ARENA, arena_id, card.nocab_uuid)
    return binder


def _uuid_for_arena_id(binder: CardBinder, arena_id: str) -> str:
    card = binder.get_by_alias(GameId.MTG, DataSource.ARENA, arena_id)
    assert card is not None
    return str(card.nocab_uuid)


class TestCombatKillInvolvementRateMetric:
    def test_hit_when_either_side_had_a_combat_kill_that_half_turn(
        self, tmp_path: Path
    ) -> None:
        """Every card in creatures_attacked/creatures_blocking that
        half-turn counts as a hit if EITHER side's
        creatures_killed_combat is non-empty - not just the card's own
        death."""
        binder = _binder_with_cards([("Owlbear", "1")])
        metric = CombatKillInvolvementRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        row = dict(_NA_ROW)
        row["user_turn_1_creatures_attacked"] = "1"
        row["user_turn_1_oppo_creatures_killed_combat"] = "2"

        metric.accumulate(row)
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = _uuid_for_arena_id(binder, "1")
        assert df.loc[owlbear_uuid, "combat_kill_involvement_rate"] == 1.0

    def test_no_hit_when_no_creature_died_in_combat_that_half_turn(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards([("Owlbear", "1")])
        metric = CombatKillInvolvementRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        row = dict(_NA_ROW)
        row["user_turn_1_creatures_attacked"] = "1"

        metric.accumulate(row)
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = _uuid_for_arena_id(binder, "1")
        assert df.loc[owlbear_uuid, "combat_kill_involvement_rate"] == 0.0

    def test_denominator_is_attackers_union_blockers(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([("Owlbear", "1"), ("Goblin Morningstar", "2")])
        metric = CombatKillInvolvementRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        row = dict(_NA_ROW)
        row["user_turn_1_creatures_attacked"] = "1"
        row["user_turn_1_creatures_blocking"] = "2"

        metric.accumulate(row)
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        uuids = set(df["nocab_uuid"])
        assert _uuid_for_arena_id(binder, "1") in uuids
        assert _uuid_for_arena_id(binder, "2") in uuids

    def test_tallies_across_both_actors_and_multiple_turns(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards([("Owlbear", "1")])
        metric = CombatKillInvolvementRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        row = dict(_NA_ROW)
        row["user_turn_1_creatures_attacked"] = "1"
        row["user_turn_1_oppo_creatures_killed_combat"] = "2"
        row["user_turn_2_creatures_attacked"] = "1"
        row["oppo_turn_1_creatures_blocking"] = "1"

        metric.accumulate(row)
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = _uuid_for_arena_id(binder, "1")
        assert df.loc[owlbear_uuid, "sample_count"] == 3

    def test_sample_count_is_total_fought_occurrences(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([("Owlbear", "1")])
        metric = CombatKillInvolvementRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        row = dict(_NA_ROW)
        row["user_turn_1_creatures_attacked"] = "1"

        metric.accumulate(row)
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = _uuid_for_arena_id(binder, "1")
        assert df.loc[owlbear_uuid, "sample_count"] == 1


class TestCombatDamagePushThroughRateMetric:
    def test_hit_is_per_card_not_turn_wide(self, tmp_path: Path) -> None:
        """Unlike CombatKillInvolvementRateMetric, only the specific
        attacker(s) in creatures_unblocked count as a hit - an
        unrelated attacker that was blocked that same half-turn does
        not."""
        binder = _binder_with_cards([("Owlbear", "1"), ("Goblin Morningstar", "2")])
        metric = CombatDamagePushThroughRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        row = dict(_NA_ROW)
        row["user_turn_1_creatures_attacked"] = "1|2"
        row["user_turn_1_creatures_unblocked"] = "1"

        metric.accumulate(row)
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
        owlbear_uuid = _uuid_for_arena_id(binder, "1")
        morningstar_uuid = _uuid_for_arena_id(binder, "2")
        assert df.loc[owlbear_uuid, "combat_damage_push_through_rate"] == 1.0
        assert df.loc[morningstar_uuid, "combat_damage_push_through_rate"] == 0.0

    def test_denominator_is_creatures_attacked_only(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([("Owlbear", "1"), ("Goblin Morningstar", "2")])
        metric = CombatDamagePushThroughRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )
        row = dict(_NA_ROW)
        # Owlbear only ever blocks (never attacks) - Goblin Morningstar
        # attacks, so the output file isn't entirely empty.
        row["user_turn_1_creatures_blocking"] = "1"
        row["user_turn_1_creatures_attacked"] = "2"

        metric.accumulate(row)
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        owlbear_uuid = _uuid_for_arena_id(binder, "1")
        assert owlbear_uuid not in set(df["nocab_uuid"])

    def test_card_never_attacking_never_appears_in_output(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([("Owlbear", "1")])
        metric = CombatDamagePushThroughRateMetric(
            binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(dict(_NA_ROW))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert df.shape[0] == 0
