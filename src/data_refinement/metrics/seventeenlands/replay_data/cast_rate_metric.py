"""CastRateMetric - BRAINSTORM.md's "Cast Rate":
P(card cast in a game | card in deck_<name>), distinct from
game_data's drawn rate since replay data can tell "drawn but never
cast" from "cast."

User-deck-only (only the user's own deck is ever fully known - see
BRAINSTORM.md's cross-cutting limitation). Scans ONLY user_turn_*
half-turns for cast events - a card in the user's own deck_<name> can
only ever be cast BY the user, so oppo_turn_* half-turns are never
relevant here (a deliberate scope narrowing distinct from
AverageTurnCastMetric's both-sides conditioning, since that metric has
no deck-membership precondition to narrow by).

A ReplayDeckHitRateMetric (replay_deck_hit_rate_metric.py) -
shares its __init__/accumulate/finalize/_rate_row with
DiscardRateMetric/TutorTargetRateMetric (replay-level) in this same
package through that base class; supplies only _hit_uuids_for_row()
and this metric's own LABEL_COLUMN/DEFAULT_OUTPUT_PATH.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_deck_hit_rate_metric import (
    ReplayDeckHitRateMetric,
)

_DEFAULT_OUTPUT_PATH = Path("data/metrics/seventeenlands/replay_data/cast_rate.parquet")


class CastRateMetric(ReplayDeckHitRateMetric):
    """Card -> P(cast at least once in the game | card in
    deck_<name>).

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str] = "cast_rate"
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def _hit_uuids_for_row(self, row: dict) -> set[UUID]:
        """Every card matched out of any user_turn_N_creatures_cast/
        non_creatures_cast cell on this row, across every N in
        self._replay_columns.user_turn_numbers.

        Inputs:
            row: one replay_data CSV row, dict-like.
        Output: the union of every matched card across every scanned
            user half-turn's creatures_cast/non_creatures_cast cell.
        Side effects: none.
        Exceptions: none expected.
        """
        card_uuids: set[UUID] = set()
        for turn in self._replay_columns.user_turn_numbers:
            card_uuids.update(
                self._replay_columns.arena_uuids(
                    row[ReplayCardColumns.turn_column("user", turn, "creatures_cast")]
                )
            )
            card_uuids.update(
                self._replay_columns.arena_uuids(
                    row[
                        ReplayCardColumns.turn_column(
                            "user", turn, "non_creatures_cast"
                        )
                    ]
                )
            )
        return card_uuids
