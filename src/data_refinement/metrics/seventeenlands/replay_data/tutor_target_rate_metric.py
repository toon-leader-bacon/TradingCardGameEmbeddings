"""TutorTargetRateMetric (replay-level) - BRAINSTORM.md's "Tutor
Target Rate (replay-level)":
P(card in user_turn_N_cards_tutored for some N | card in deck_<name>).

Distinct class from game_data.tutor_target_rate_metric.TutorTargetRateMetric
- same name reused deliberately, since each lives in its own
source-specific package (the same convention every repeated concept
name across draft_data/game_data already follows). User-only by
construction: there is no oppo_turn_N_cards_tutored column at all
(opponent tutoring is a hidden-library action, confirmed by the
original BRAINSTORM.md pass) - unlike DiscardRateMetric/CastRateMetric,
this isn't a scope choice, it's the only column that exists.

A ReplayDeckHitRateMetric (replay_deck_hit_rate_metric.py) - shares
its __init__/accumulate/finalize/_rate_row with cast_rate_metric.py's
CastRateMetric/discard_rate_metric.py's DiscardRateMetric through that
base class; supplies only _hit_uuids_for_row() and this metric's own
LABEL_COLUMN/DEFAULT_OUTPUT_PATH.
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

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/tutor_target_rate.parquet"
)


class TutorTargetRateMetric(ReplayDeckHitRateMetric):
    """Card -> P(tutored at least once in the game | card in
    deck_<name>).

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str] = "tutor_target_rate"
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def _hit_uuids_for_row(self, row: dict) -> set[UUID]:
        """Every card matched out of any user_turn_N_cards_tutored
        cell on this row, across every N in
        self._replay_columns.user_turn_numbers.

        Inputs:
            row: one replay_data CSV row, dict-like.
        Output: the union of every matched card across every scanned
            user half-turn's cards_tutored cell.
        Side effects: none.
        Exceptions: none expected.
        """
        card_uuids: set[UUID] = set()
        for turn in self._replay_columns.user_turn_numbers:
            card_uuids.update(
                self._replay_columns.arena_uuids(
                    row[ReplayCardColumns.turn_column("user", turn, "cards_tutored")]
                )
            )
        return card_uuids
