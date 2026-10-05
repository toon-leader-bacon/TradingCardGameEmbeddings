"""DiscardRateMetric - BRAINSTORM.md's "Discard Rate":
P(card in cards_discarded (its own side's discards) in some turn |
card in deck_<name>).

User-deck-only, same shape and scope-narrowing reasoning as
cast_rate_metric.py's CastRateMetric (scans only user_turn_* half-
turns - a discard, like a cast, can only ever be this user's own deck
card). A ReplayDeckHitRateMetric (replay_deck_hit_rate_metric.py) -
shares its __init__/accumulate/finalize/_rate_row with
CastRateMetric/TutorTargetRateMetric (replay-level) in this same
package through that base class; supplies only _hit_uuids_for_row()
and this metric's own LABEL_COLUMN/DEFAULT_OUTPUT_PATH.

OPEN CAVEAT (carried forward from BRAINSTORM.md, not settled here):
whether a
user_turn_N_cards_discarded cell can ever reflect a FORCED discard
(e.g. an opposing discard spell) rather than only a self-inflicted
cleanup discard is unconfirmed - this metric's docstring states this
as a labelled assumption, not a verified fact, and design-recipe-
implement should not silently settle the ambiguity either way without
flagging it back.
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
    "data/metrics/seventeenlands/replay_data/discard_rate.parquet"
)


class DiscardRateMetric(ReplayDeckHitRateMetric):
    """Card -> P(discarded at least once in the game | card in
    deck_<name>).

    ASSUMPTION, not verified this round: every user_turn_N_cards_discarded
    hit reflects a self-inflicted discard (hand size / cleanup), never
    a forced discard from an opposing effect that hit this user during
    their own turn - see module docstring's "OPEN CAVEAT".

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str] = "discard_rate"
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def _hit_uuids_for_row(self, row: dict) -> set[UUID]:
        """Every card matched out of any user_turn_N_cards_discarded
        cell on this row, across every N in
        self._replay_columns.user_turn_numbers.

        Inputs:
            row: one replay_data CSV row, dict-like.
        Output: the union of every matched card across every scanned
            user half-turn's cards_discarded cell.
        Side effects: none.
        Exceptions: none expected.
        """
        card_uuids: set[UUID] = set()
        for turn in self._replay_columns.user_turn_numbers:
            card_uuids.update(
                self._replay_columns.arena_uuids(
                    row[ReplayCardColumns.turn_column("user", turn, "cards_discarded")]
                )
            )
        return card_uuids
