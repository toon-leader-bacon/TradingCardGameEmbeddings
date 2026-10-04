"""AverageTurnCastMetric - BRAINSTORM.md's "Average Turn Cast": per card,
the average turn number it is cast on, over every occurrence in
creatures_cast or non_creatures_cast, either actor, no deck condition.

A CodeCountTableMetric (code_count_table_metric.py): partitions hold
(count, turn_sum) per card; the average is taken when a slice is built.
Every matched entry counts, so two copies cast in one half-turn count
twice, as the row implementation did.
"""

from typing import ClassVar

import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import ratio_output
from src.data_refinement.metrics.seventeenlands.replay_data.code_count_table_metric import (
    CodeCountTableMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    CAST_FIELDS,
    ReplayDataChunk,
)

COUNT_COLUMN = "count"
TURN_SUM_COLUMN = "turn_sum"


class AverageTurnCastMetric(CodeCountTableMetric):
    """Card -> average turn number it is cast on."""

    OUTPUT_STEM: ClassVar[str] = "average_turn_cast"
    LABEL_COLUMN: ClassVar[str] = "average_turn_cast"
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (COUNT_COLUMN, TURN_SUM_COLUMN)

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): turn_sum / count,
        with count as sample_count.

        Inputs: summed (nocab_uuid, count, turn_sum), baseline (None).
        Output: nocab_uuid, average_turn_cast, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> AverageTurnCastMetric.output_from_counts(summed, None)
        """
        return ratio_output(
            summed, cls.KEY_COLUMNS, TURN_SUM_COLUMN, COUNT_COLUMN, cls.LABEL_COLUMN
        )

    def _tally(self, chunk: ReplayDataChunk) -> None:
        """See CodeCountTableMetric._tally(): every matched cast entry,
        counted and summed by turn.

        Inputs: chunk. Output: none.
        Side effects: adds to the tallies. Exceptions: none.
        """
        cast = chunk.events_for(CAST_FIELDS).matched()
        self._tallies.add(0, cast.codes)
        self._tallies.add(1, cast.codes, cast.turns)
