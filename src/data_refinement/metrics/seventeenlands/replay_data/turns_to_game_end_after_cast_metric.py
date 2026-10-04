"""TurnsToGameEndAfterCastMetric - BRAINSTORM.md's "Turns to Game End
After Cast": per card, the average of num_turns minus the first turn it
was cast that game (either actor), over every game it was cast in.

num_turns is on the same per-actor turn scale as user_turn_N/oppo_turn_N
(checked against MSH.PremierDraft), so no unit conversion is needed.
"First" is the smallest turn number seen on either actor's counter, not
a globally ordered occurrence: actor counters are not one shared elapsed
index.

A CodeCountTableMetric (code_count_table_metric.py): partitions hold
(count, delta_sum) per card, one entry per (game, card cast that game).
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt
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
DELTA_SUM_COLUMN = "delta_sum"


class TurnsToGameEndAfterCastMetric(CodeCountTableMetric):
    """Card -> average (num_turns - first turn cast)."""

    OUTPUT_STEM: ClassVar[str] = "turns_to_game_end_after_cast"
    LABEL_COLUMN: ClassVar[str] = "turns_to_game_end_after_cast"
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (COUNT_COLUMN, DELTA_SUM_COLUMN)

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): delta_sum / count,
        with count as sample_count.

        Inputs: summed (nocab_uuid, count, delta_sum), baseline (None).
        Output: nocab_uuid, turns_to_game_end_after_cast, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> TurnsToGameEndAfterCastMetric.output_from_counts(summed, None)
        """
        return ratio_output(
            summed, cls.KEY_COLUMNS, DELTA_SUM_COLUMN, COUNT_COLUMN, cls.LABEL_COLUMN
        )

    def _tally(self, chunk: ReplayDataChunk) -> None:
        """See CodeCountTableMetric._tally(): per (game, card cast that
        game), num_turns minus the card's first cast turn.

        Inputs: chunk. Output: none.
        Side effects: adds to the tallies. Exceptions: none.
        """
        rows, codes, first_turns = _first_cast_turns(chunk)
        self._tallies.add(0, codes)
        self._tallies.add(1, codes, chunk.num_turns[rows] - first_turns)


def _first_cast_turns(
    chunk: ReplayDataChunk,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int16]]:
    """Each (row, card) cast that game, and its smallest cast turn on
    either actor.

    Inputs: chunk.
    Output: (rows, codes, first turns), one entry per distinct (row,
        card).
    Side effects: none. Exceptions: none.
    """
    cast = chunk.events_for(CAST_FIELDS).matched()

    # Sort by (row, card), then turn: each pair's first entry is its first cast
    pair_keys = cast.rows.astype(np.int64) * len(chunk.card_uuids) + cast.codes
    order = np.lexsort((cast.turns, pair_keys))
    _, firsts = np.unique(pair_keys[order], return_index=True)
    first_entries = order[firsts]
    return (
        cast.rows[first_entries],
        cast.codes[first_entries],
        cast.turns[first_entries],
    )
