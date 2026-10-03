"""PickNumberDecayCurveMetric - BRAINSTORM.md's single-card metric #7
(Pick-Number Decay Curve): per card, its take rate at every pick_number,
as one vector.

A PackCardTallyMetric (pack_card_tally_metric.py) keyed by (card,
pick_number). Partitions hold that long form, (in_pack, picked) per
(card, pick_number); output_from_counts regroups a slice into one row
per card with two parallel lists, the shape
PickNumberDecayCurveDataConstructor reads.

MAX_BUCKET_COUNT clamps every pack's tail: pick_number >=
MAX_BUCKET_COUNT - 1 folds into the last bucket rather than growing it
(a 20-card pack's picks 14-19 all tally into bucket 14), so the dojo's
fixed-width vector never loses a format's late picks.

BUCKET COUNT IS DERIVED: a slice's lists are max(pick_number) + 1 long,
the largest bucket actually seen in that slice, never a hardcoded pack
size. An unseen bucket has sample count 0 and a null take rate.
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    DraftDataChunk,
    DraftStratum,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metric import (
    IN_PACK_COLUMN,
    PICKED_COLUMN,
    PackCardTallyMetric,
)

TAKE_RATE_COLUMN = "take_rate_by_pick_number"
SAMPLE_COUNT_COLUMN = "sample_count_by_pick_number"


class PickNumberDecayCurveMetric(PackCardTallyMetric):
    """Card -> take rate, as a vector indexed by pick_number."""

    OUTPUT_STEM: ClassVar[str] = "pick_number_decay_curve"
    LABEL_COLUMN: ClassVar[str] = TAKE_RATE_COLUMN
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("nocab_uuid", "pick_number")
    MAX_BUCKET_COUNT: ClassVar[int] = 15

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): one row per card,
        with take_rate_by_pick_number (list[float | None]) and
        sample_count_by_pick_number (list[int]), each max(pick_number)
        + 1 long over the slice.

        Inputs: summed (nocab_uuid, pick_number, in_pack, picked),
            baseline (None).
        Output: nocab_uuid, take_rate_by_pick_number,
            sample_count_by_pick_number; cards sorted by uuid.
        Side effects: none. Exceptions: none.

        Example:
            >>> PickNumberDecayCurveMetric.output_from_counts(summed, None)
        """
        cards = summed.column("nocab_uuid").to_pylist()
        pick_numbers = summed.column("pick_number").to_numpy()
        in_pack = summed.column(IN_PACK_COLUMN).to_numpy()
        picked = summed.column(PICKED_COLUMN).to_numpy()
        bucket_count = int(pick_numbers.max()) + 1 if len(cards) else 0

        # One (bucket_count,) count vector per card
        distinct_cards = sorted(set(cards))
        card_index = {card: index for index, card in enumerate(distinct_cards)}
        rows = np.array([card_index[card] for card in cards], np.intp)
        in_pack_by_card = np.zeros((len(distinct_cards), bucket_count), np.float64)
        picked_by_card = np.zeros_like(in_pack_by_card)
        np.add.at(in_pack_by_card, (rows, pick_numbers), in_pack)
        np.add.at(picked_by_card, (rows, pick_numbers), picked)

        return pa.table(
            {
                "nocab_uuid": pa.array(distinct_cards, pa.string()),
                TAKE_RATE_COLUMN: pa.array(
                    [_rates(n, d) for n, d in zip(picked_by_card, in_pack_by_card)],
                    pa.list_(pa.float64()),
                ),
                SAMPLE_COUNT_COLUMN: pa.array(
                    [np.rint(d).astype(np.int64).tolist() for d in in_pack_by_card],
                    pa.list_(pa.int64()),
                ),
            }
        )

    def _stratum_values(
        self, chunk: DraftDataChunk, stratum: DraftStratum
    ) -> npt.NDArray[np.generic]:
        """See PackCardTallyMetric._stratum_values(): pick_number,
        clamped to MAX_BUCKET_COUNT - 1.

        Inputs: chunk, stratum (always PICK_NUMBER here).
        Output: int64 array (rows,).
        Side effects: none. Exceptions: none.
        """
        return np.minimum(chunk.stratum(stratum), self.MAX_BUCKET_COUNT - 1)


def _rates(
    picked: npt.NDArray[np.float64], in_pack: npt.NDArray[np.float64]
) -> list[float | None]:
    """One card's take rate per bucket: picked / in_pack, None where
    in_pack is 0.

    Inputs: picked, in_pack (one card's per-bucket counts).
    Output: list[float | None], one per bucket.
    Side effects: none. Exceptions: none.
    """
    return [
        None if total == 0 else float(taken / total)
        for taken, total in zip(picked, in_pack)
    ]
