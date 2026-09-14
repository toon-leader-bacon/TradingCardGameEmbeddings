"""Concrete ReplayTurnEventRateMetric subclasses - see
plans/replay_data_metrics.md's Component overview.
"""

from pathlib import Path
from typing import ClassVar, Literal
from uuid import UUID

import pandas as pd

from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metric import (
    ReplayTurnEventRateMetric,
)

_COMBAT_KILL_INVOLVEMENT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/combat_kill_involvement_rate.parquet"
)
_COMBAT_DAMAGE_PUSH_THROUGH_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/combat_damage_push_through_rate.parquet"
)


class CombatKillInvolvementRateMetric(ReplayTurnEventRateMetric):
    """Card -> P(a creature died in combat that half-turn | card fought
    - attacked or blocked - that half-turn).

    Not "this card died" - "a creature (either side's) died as a
    result of a half-turn this card fought in." See
    _numerator_cards_for_turn()'s docstring for the turn-wide-boolean
    shape this implies.
    """

    LABEL_COLUMN: ClassVar[str] = "combat_kill_involvement_rate"
    DEFAULT_OUTPUT_PATH = _COMBAT_KILL_INVOLVEMENT_OUTPUT_PATH

    def _denominator_cards_for_turn(
        self, row: dict, actor: Literal["user", "oppo"], turn: int
    ) -> list[UUID]:
        """See ReplayTurnEventRateMetric._denominator_cards_for_turn().
        Cards in this half-turn's creatures_attacked union
        creatures_blocking."""
        attacked = self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "creatures_attacked")]
        )
        blocking = self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "creatures_blocking")]
        )
        return list(set(attacked) | set(blocking))

    def _numerator_cards_for_turn(
        self,
        row: dict,
        actor: Literal["user", "oppo"],
        turn: int,
        denominator_cards: list[UUID],
    ) -> set[UUID]:
        """See ReplayTurnEventRateMetric._numerator_cards_for_turn().
        The full denominator_cards set if either side's
        creatures_killed_combat is non-empty this half-turn, else the
        empty set - a turn-wide boolean applied uniformly, not a
        per-card check.

        Checks the RAW cell's presence (pd.notna), not
        arena_uuids()'s matched output - a creature that died but
        doesn't match a known card (e.g. not yet in the binder) still
        means a kill happened; this is a "did anything die" signal,
        not an identity lookup."""
        user_killed = row[
            ReplayCardColumns.turn_column(actor, turn, "user_creatures_killed_combat")
        ]
        oppo_killed = row[
            ReplayCardColumns.turn_column(actor, turn, "oppo_creatures_killed_combat")
        ]
        if pd.notna(user_killed) or pd.notna(oppo_killed):
            return set(denominator_cards)
        return set()


class CombatDamagePushThroughRateMetric(ReplayTurnEventRateMetric):
    """Card -> P(card appears in creatures_unblocked | card appears in
    creatures_attacked, same half-turn) - how often this attacker's
    damage actually connects."""

    LABEL_COLUMN: ClassVar[str] = "combat_damage_push_through_rate"
    DEFAULT_OUTPUT_PATH = _COMBAT_DAMAGE_PUSH_THROUGH_OUTPUT_PATH

    def _denominator_cards_for_turn(
        self, row: dict, actor: Literal["user", "oppo"], turn: int
    ) -> list[UUID]:
        """See ReplayTurnEventRateMetric._denominator_cards_for_turn().
        Cards in this half-turn's creatures_attacked."""
        return self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "creatures_attacked")]
        )

    def _numerator_cards_for_turn(
        self,
        row: dict,
        actor: Literal["user", "oppo"],
        turn: int,
        denominator_cards: list[UUID],
    ) -> set[UUID]:
        """See ReplayTurnEventRateMetric._numerator_cards_for_turn().
        denominator_cards intersected with this half-turn's
        creatures_unblocked - a genuine per-card subset."""
        unblocked = self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "creatures_unblocked")]
        )
        return set(denominator_cards) & set(unblocked)
