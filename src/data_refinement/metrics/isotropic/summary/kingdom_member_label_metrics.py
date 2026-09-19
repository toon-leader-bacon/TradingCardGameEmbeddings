"""Every concrete KingdomMemberLabelMetric (kingdom_member_label_metric.py)
this container has today - BRAINSTORM.md's "New candidates" section
#12 and #13, the in/out and count-valued framings of "given the
kingdom, what does the winner's deck do with each card."
"""

from pathlib import Path

import pyarrow as pa

from src.data_refinement.metrics.isotropic.summary.kingdom_member_label_metric import (
    KingdomMemberLabelMetric,
)


class WinningDeckMembershipMetric(KingdomMemberLabelMetric):
    """Kingdom card -> is it in the winner's end.deck at all
    (BRAINSTORM.md's "New candidates" section #12)."""

    LABEL_COLUMN = "in_winning_deck"
    LABEL_TYPE = pa.bool_()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/isotropic/winning_deck_membership.parquet")

    def _label_for_kingdom_card(
        self, card_name: str, winner_end_deck: dict[str, int]
    ) -> bool:
        return card_name in winner_end_deck


class WinningDeckCountMetric(KingdomMemberLabelMetric):
    """Kingdom card -> how many copies are in the winner's end.deck
    (0 if absent) (BRAINSTORM.md's "New candidates" section #13)."""

    LABEL_COLUMN = "winning_deck_count"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/isotropic/winning_deck_count.parquet")

    def _label_for_kingdom_card(
        self, card_name: str, winner_end_deck: dict[str, int]
    ) -> int:
        return winner_end_deck.get(card_name, 0)
