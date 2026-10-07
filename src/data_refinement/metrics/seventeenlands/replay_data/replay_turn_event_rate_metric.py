"""Template Method base for per-card hit rates over half-turns: per card,
(times in a half-turn's denominator, times also a hit) (see
replay_data/README.md).

A CodeCountTableMetric (code_count_table_metric.py): partitions hold
(total, hits) per card; the rate (hits / total) is taken when a slice is
built. A subclass fixes OUTPUT_STEM and LABEL_COLUMN and implements:

- _denominator_and_hit_codes(chunk): one code per denominator entry,
  across every half-turn of the chunk (a card counted twice in one
  half-turn, if the subclass's denominator keeps duplicates, counts
  twice), and one code per hit.

Each list is tallied with one np.bincount; which entries count, and
whether duplicates within a half-turn collapse, is each subclass's own
rule, kept exactly as the row implementation counted.
"""

from abc import abstractmethod
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import ratio_output
from src.data_refinement.metrics.seventeenlands.replay_data.code_count_table_metric import (
    CodeCountTableMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    ReplayDataChunk,
)

TOTAL_COLUMN = "total"
HITS_COLUMN = "hits"


class ReplayTurnEventRateMetric(CodeCountTableMetric):
    """Per card: hits / total over half-turns."""

    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (TOTAL_COLUMN, HITS_COLUMN)

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): hits / total as
        LABEL_COLUMN, total as sample_count.

        Inputs: summed (nocab_uuid, total, hits), baseline (None).
        Output: nocab_uuid, LABEL_COLUMN, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> CombatKillInvolvementRateMetric.output_from_counts(summed, None)
        """
        return ratio_output(
            summed, cls.KEY_COLUMNS, HITS_COLUMN, TOTAL_COLUMN, cls.LABEL_COLUMN
        )

    def _tally(self, chunk: ReplayDataChunk) -> None:
        """See CodeCountTableMetric._tally(): the denominator entries
        into total, the hit entries into hits.

        Inputs: chunk. Output: none.
        Side effects: adds to the tallies. Exceptions: none.
        """
        denominator, hits = self._denominator_and_hit_codes(chunk)
        self._tallies.add(0, denominator)
        self._tallies.add(1, hits)

    @abstractmethod
    def _denominator_and_hit_codes(
        self, chunk: ReplayDataChunk
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """One card code per denominator entry, and one per hit, over the
        chunk.

        Inputs: chunk. Output: (denominator codes, hit codes), int32, >= 0.
        Side effects: none. Exceptions: implementation-defined.
        """
        raise NotImplementedError
