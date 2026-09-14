"""PickNumberDecayCurveMetric - BRAINSTORM.md's single-card metric #7
(Pick-Number Decay Curve): for a card, P(picked | in pack) as a
function of pick_number within its pack - one output row per CARD,
carrying a pick_number-indexed vector, rather than
pack_card_tally_metrics.py's one-row-per-key shape.

Still a PackCardTallyMetric (pack_card_tally_metric.py) subclass -
reuses accumulate()/_tally_key() unchanged (KEY_COLUMNS =
("pick_number",)), so every row still tallies into the same
self._times_in_pack/_times_picked dicts as every other
PackCardTallyMetric subclass. Only finalize() differs: instead of one
row per (card, pick_number) key, it collects every pick_number bucket
tallied for a given card into that card's own row, as parallel
take_rate_by_pick_number/sample_count_by_pick_number list columns - the
same "parallel lists, one row per card" convention
sts_gg/card_character_prediction_metric.py already established for a
similarly-shaped per-card frequency table. Kept in its own file (rather
than folded into pack_card_tally_metrics.py) for the same reason
sts_gg/ascension_prediction_metric.py is kept separate from
deck_label_metrics.py - a distinct enough override to warrant it.

BUCKET COUNT IS DERIVED, NOT HARDCODED: the vector length for every
output row is max(pick_number) + 1 across every key this instance ever
tallied - never a hardcoded pack size (14 for MSH.PremierDraft, but
BRAINSTORM.md explicitly flags this as varying by set/event_type) -
see plans/draft_data_metrics.md's "Data facts" section (the same
lesson BRAINSTORM.md draws for Wheel Rate's table_size, applied here
even though this container doesn't build Wheel Rate).
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metric import (
    PackCardTallyMetric,
)


class PickNumberDecayCurveMetric(PackCardTallyMetric):
    """Card -> take rate, as a vector indexed by pick_number."""

    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("pick_number",)
    DEFAULT_OUTPUT_PATH = Path(
        "data/metrics/seventeenlands/draft_data/pick_number_decay_curve.parquet"
    )

    def _tally_key(self, row: dict, card_uuid: UUID) -> tuple:
        """See PackCardTallyMetric._tally_key(). Key = (card_uuid,
        row["pick_number"])."""
        return (card_uuid, row["pick_number"])

    def finalize(self) -> Path:
        """Build one row per card, bucketing every tallied pick_number
        into that card's own take_rate_by_pick_number/
        sample_count_by_pick_number lists, and write self._output_path.

        Overrides PackCardTallyMetric.finalize() entirely - see module
        docstring for why this metric's output shape can't reuse the
        shared per-key finalize().

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, take_rate_by_pick_number:
            list[float], sample_count_by_pick_number: list[int] - one
            row per card seen at least once, each list's length equal
            to the observed max pick_number + 1).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/draft_data/pick_number_decay_curve.parquet')
        """
        bucket_count = self._bucket_count()
        cards = self._distinct_cards()

        rows: list[dict] = []
        # Build one row per card, filling in every pick_number bucket
        # (an unseen bucket gets sample_count 0 and a null take_rate).
        for card_uuid in cards:
            rows.append(self._decay_curve_row(card_uuid, bucket_count))

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, self._output_path)
        return self._output_path

    def _bucket_count(self) -> int:
        """The observed max pick_number + 1 across every tallied key.

        Private helper - single consumer is finalize(). See module
        docstring's BUCKET COUNT IS DERIVED note.

        Inputs: none (uses accumulated state).
        Output: max pick_number + 1 across self._times_in_pack's keys,
            or 0 if nothing was ever tallied.
        Side effects: none.
        Exceptions: none.
        """
        pick_numbers = [key[1] for key in self._times_in_pack]
        return max(pick_numbers) + 1 if pick_numbers else 0

    def _distinct_cards(self) -> list[UUID]:
        """Every distinct card_uuid tallied at least once.

        Private helper - single consumer is finalize().

        Inputs: none (uses accumulated state).
        Output: every distinct key[0] across self._times_in_pack, order
            not guaranteed.
        Side effects: none.
        Exceptions: none.
        """
        return list({key[0] for key in self._times_in_pack})

    def _decay_curve_row(self, card_uuid: UUID, bucket_count: int) -> dict:
        """Build one card's output row: its take-rate/sample-count
        vectors across every pick_number bucket in [0, bucket_count).

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: the card this row is for.
            bucket_count: total pick_number buckets, from
                self._bucket_count().
        Output: a dict with keys "nocab_uuid" (str, card_uuid),
            "take_rate_by_pick_number" (list[float | None], length
            bucket_count - None where sample_count is 0 at that
            bucket), "sample_count_by_pick_number" (list[int], same
            length).
        Side effects: none.
        Exceptions: none.
        """
        take_rate_by_pick_number: list[float | None] = []
        sample_count_by_pick_number: list[int] = []
        for pick_number in range(bucket_count):
            key = (card_uuid, pick_number)
            times_in_pack = self._times_in_pack.get(key, 0)
            sample_count_by_pick_number.append(times_in_pack)
            if times_in_pack == 0:
                take_rate_by_pick_number.append(None)
            else:
                times_picked = self._times_picked.get(key, 0)
                take_rate_by_pick_number.append(times_picked / times_in_pack)

        return {
            "nocab_uuid": str(card_uuid),
            "take_rate_by_pick_number": take_rate_by_pick_number,
            "sample_count_by_pick_number": sample_count_by_pick_number,
        }
