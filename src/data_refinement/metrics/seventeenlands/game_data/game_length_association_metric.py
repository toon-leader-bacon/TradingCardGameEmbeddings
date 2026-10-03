"""GameLengthAssociationMetric - BRAINSTORM.md's single-card metric
"Game Length Association": for a card, the average num_turns across
every game it was in deck_<name>, minus the average num_turns across
every game in the slice. A signed aggro (negative) /
control (positive) curve-sensitivity proxy.

A GameCardAverageMetric (game_card_average_metric.py) with a baseline
(HAS_BASELINE), using the base's three optional Template Method steps:

- _extra_accumulate(chunk) adds every row to this partition's
  (turn sum, game count) baseline, ungated by any card;
- _baseline_counts() writes it as the partition's null-key baseline row
  (games = game count, value_sum = turn sum);
- output_from_counts(summed, baseline) returns each card's average minus
  the slice's baseline mean (the baseline rows summed over the slice).

The bases' accumulate() and finalize() are reused unchanged.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metric import (
    GAMES_COLUMN,
    VALUE_SUM_COLUMN,
    GameCardAverageMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata


class GameLengthAssociationMetric(GameCardAverageMetric):
    """Card -> (average num_turns when in deck) - (the slice's average
    num_turns)."""

    OUTPUT_STEM: ClassVar[str] = "game_length_association"
    LABEL_COLUMN: ClassVar[str] = "game_length_association"
    HAS_BASELINE: ClassVar[bool] = True
    ZONE: ClassVar[GameZone] = GameZone.DECK

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path,
    ) -> None:
        """Same parameters as GameCardAverageMetric; adds the format-wide
        baseline tallies.

        Inputs: see GameCardAverageMetric.__init__().
        Output: none (constructor).
        Side effects: as the base, plus zeroed baseline tallies.
        Exceptions: none.
        """
        super().__init__(version_metadata, output_path)
        self._global_turn_sum = 0.0
        self._global_game_count = 0

    def _values(self, chunk: GameDataChunk) -> npt.NDArray[np.float64]:
        """See GameCardAverageMetric._values(): num_turns as float.

        Inputs: chunk. Output: shape (len(chunk),).
        Side effects: none. Exceptions: none.
        """
        return chunk.num_turns.astype(np.float64)

    def _extra_accumulate(self, chunk: GameDataChunk) -> None:
        """Add every row of chunk to the format-wide baseline.

        Inputs: chunk. Output: none.
        Side effects: updates _global_turn_sum / _global_game_count.
        Exceptions: none.
        """
        self._global_turn_sum += float(chunk.num_turns.sum(dtype=np.float64))
        self._global_game_count += len(chunk)

    def _baseline_counts(self) -> npt.NDArray[np.float64] | None:
        """See CardCountTableMetric._baseline_counts(): this
        partition's game count and turn sum, in COUNT_COLUMNS order.

        Inputs: none.
        Output: [game count, turn sum] (float64).
        Side effects: none. Exceptions: none.
        """
        return np.array(
            [self._global_game_count, self._global_turn_sum], dtype=np.float64
        )

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): each card's
        average num_turns minus the slice's (baseline value_sum /
        baseline games).

        Inputs: summed (nocab_uuid, games, value_sum), baseline (one
            row of games, value_sum).
        Output: nocab_uuid, LABEL_COLUMN, sample_count.
        Side effects: none.
        Exceptions: ValueError if baseline is None, or not one row, or has
            zero games.

        Example:
            >>> GameLengthAssociationMetric.output_from_counts(summed, baseline)
        """
        # Validate: a baseline with at least one game
        if baseline is None:
            raise ValueError(f"{cls.__name__} needs the slice's baseline")
        baseline_mean = _baseline_mean(baseline)

        # Each card's average, then shifted by the slice's mean
        result = super().output_from_counts(summed, None)
        return _subtract_from_label(result, cls.LABEL_COLUMN, baseline_mean)


def _baseline_mean(baseline: pa.Table) -> float:
    """baseline's value_sum / games.

    Inputs: baseline, one row. Output: float.
    Side effects: none.
    Exceptions: ValueError if baseline is not one row or has zero games.
    """
    if baseline.num_rows != 1:
        raise ValueError(f"expected a one-row baseline, got {baseline.num_rows} rows")
    games = float(baseline.column(GAMES_COLUMN)[0].as_py())
    if games <= 0:
        raise ValueError("the slice's baseline has no games")
    return float(baseline.column(VALUE_SUM_COLUMN)[0].as_py()) / games


def _subtract_from_label(table: pa.Table, label_column: str, amount: float) -> pa.Table:
    """table with amount subtracted from every label_column value.

    Inputs: table, label_column, amount. Output: pa.Table.
    Side effects: none. Exceptions: KeyError if label_column is missing.
    """
    index = table.schema.get_field_index(label_column)
    if index < 0:
        raise KeyError(label_column)
    shifted = pc.subtract(table.column(index), amount)
    return table.set_column(index, label_column, shifted)
