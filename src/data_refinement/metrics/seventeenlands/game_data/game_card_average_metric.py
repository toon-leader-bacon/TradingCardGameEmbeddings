"""Template Method base for per-card metrics that average one per-game
scalar over every game a card was present (count > 0) in, in one
game_data zone (see game_data/README.md).

A GameCardCountTableMetric (card_count_table_metric.py): each partition
holds, per card, the raw (games, value_sum) counts; the average is taken
only when a slice is built (output_from_counts), so any slice's average
is exact.

A subclass fixes LABEL_COLUMN / OUTPUT_STEM / ZONE and implements
_values(); GameLengthAssociationMetric also uses the base's optional
_extra_accumulate() and _baseline_counts() steps and overrides
output_from_counts() to subtract the slice's baseline mean.

Tallies are kept per matched column (CardColumnTallies) and grouped by
card only in finalize(), so two header columns naming one card both
count, exactly as the row implementation did. "games" is the first
count column, the denominator.
"""

from abc import abstractmethod
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import ratio_output
from src.data_refinement.metrics.seventeenlands.game_data.card_count_table_metric import (
    GameCardCountTableMetric,
)
from src.data_refinement.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)

GAMES_COLUMN = "games"
VALUE_SUM_COLUMN = "value_sum"


class GameCardAverageMetric(GameCardCountTableMetric):
    """Per-card average of one per-game scalar, across every game the
    card was present in, in this subclass's ZONE."""

    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (GAMES_COLUMN, VALUE_SUM_COLUMN)

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): each card's
        value_sum / games as LABEL_COLUMN, games as sample_count.

        Inputs: summed (nocab_uuid, games, value_sum), baseline (unused
            here; None).
        Output: nocab_uuid, LABEL_COLUMN, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> DrawnWinRateMetric.output_from_counts(summed, None)
        """
        return ratio_output(
            summed, cls.KEY_COLUMNS, VALUE_SUM_COLUMN, GAMES_COLUMN, cls.LABEL_COLUMN
        )

    @abstractmethod
    def _values(self, chunk: GameDataChunk) -> npt.NDArray[np.float64]:
        """The scalar to average, one per row of chunk.

        Inputs: chunk. Output: shape (len(chunk),).
        Side effects: none expected. Exceptions: implementation-defined.
        """
        raise NotImplementedError

    def _increments(self, chunk: GameDataChunk) -> npt.NDArray[np.float64]:
        """See CardCountTableMetric._increments(): per column, the games
        its card was present in, and the sum of _values() over them.

        Inputs: chunk.
        Output: shape (2, columns): present summed over rows, and
            present^T @ _values(chunk).
        Side effects: none.
        Exceptions: ValueError if _values()' length differs from the
            chunk's row count.
        """
        present = chunk.zones[self.ZONE].present()
        values = self._values(chunk)
        if values.shape[0] != present.shape[0]:
            raise ValueError(
                f"{type(self).__name__}: {values.shape[0]} values for "
                f"{present.shape[0]} rows"
            )
        return np.stack(
            [
                present.sum(axis=0, dtype=np.float64),
                present.T.astype(np.float64) @ values,
            ]
        )
