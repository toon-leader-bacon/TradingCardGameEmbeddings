"""CardMeanTally - the running per-card (value sum, sample count) that the
accumulation metrics fold rows into, and the mean table they write.

Shared by the dict-based per-card metrics (sts_gg, sts2_runs and
isotropic): each held two dicts keyed by card uuid and divided them in
finalize(). A rate is the mean of a 0/1 indicator, so rate metrics add
1.0 or 0.0 per sample.

The 17lands metrics keep the same shape as vectorized count tables
(seventeenlands/count_table.py) instead, and the per-card distribution
metrics (e.g. sts_gg/card_character_prediction_metric.py) keep one count
per (card, class), so neither uses this.
"""

from uuid import UUID

import pandas as pd

NOCAB_UUID_COLUMN = "nocab_uuid"
SAMPLE_COUNT_COLUMN = "sample_count"


class CardMeanTally:
    """Per-card running sum and count of one value."""

    def __init__(self) -> None:
        """Start with no cards.

        Inputs: none. Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        self._value_sum: dict[UUID, float] = {}
        self._sample_count: dict[UUID, int] = {}

    def add(self, card_uuid: UUID, value: float) -> None:
        """Count one sample of value for card_uuid.

        Inputs: card_uuid (UUID); value (float; a bool or int is fine:
            it is summed).
        Output: none.
        Side effects: adds value to the card's sum and 1 to its count.
        Exceptions: none.

        Example:
            >>> tally.add(card_uuid, 1.0)
        """
        self._value_sum[card_uuid] = self._value_sum.get(card_uuid, 0.0) + value
        self._sample_count[card_uuid] = self._sample_count.get(card_uuid, 0) + 1

    def mean_frame(self, label_column: str) -> pd.DataFrame:
        """One row per card added at least once, in first-added order.

        Inputs:
            label_column: the name of the mean column.
        Output: columns nocab_uuid (str), label_column (float: the
            card's value sum / sample count) and sample_count
            (int). No rows, but the same columns, when nothing was added.
        Side effects: none. Exceptions: none.

        Example:
            >>> tally.add(card_uuid, 1.0)
            >>> tally.add(card_uuid, 0.0)
            >>> tally.mean_frame("win_rate").loc[0, "win_rate"]
            0.5
        """
        rows = [
            {
                NOCAB_UUID_COLUMN: str(card_uuid),
                label_column: self._value_sum[card_uuid] / count,
                SAMPLE_COUNT_COLUMN: count,
            }
            for card_uuid, count in self._sample_count.items()
        ]
        frame = pd.DataFrame(
            rows, columns=[NOCAB_UUID_COLUMN, label_column, SAMPLE_COUNT_COLUMN]
        )
        # An empty frame's columns would otherwise all be object dtype
        return frame.astype({label_column: "float64", SAMPLE_COUNT_COLUMN: "int64"})
