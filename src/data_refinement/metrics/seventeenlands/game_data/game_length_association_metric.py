"""GameLengthAssociationMetric - BRAINSTORM.md's single-card metric
"Game Length Association": for a card, the average num_turns across
every game it was in deck_<name>, minus the format-wide average
num_turns across every game scanned. A signed aggro (negative) /
control (positive) curve-sensitivity proxy.

A GameCardAverageMetric (game_card_average_metric.py) that uses both of
the base's optional Template Method steps, and nothing else:

- _extra_accumulate(chunk) adds every row to a format-wide
  (turn sum, game count) baseline, ungated by any card;
- _label(value_sum, count) returns the card's average minus that
  baseline.

The base's accumulate() and finalize() are reused unchanged.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metric import (
    GameCardAverageMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/game_length_association.parquet"
)


class GameLengthAssociationMetric(GameCardAverageMetric):
    """Card -> (average num_turns when in deck) - (format-wide average
    num_turns)."""

    LABEL_COLUMN: ClassVar[str] = "game_length_association"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_PATH
    ZONE: ClassVar[GameZone] = GameZone.DECK

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path | None = None,
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

    def _label(self, value_sum: float, count: int) -> float:
        """The card's average num_turns minus the format-wide average.

        Inputs: value_sum, count (count >= 1). Output: float.
        Side effects: none.
        Exceptions: none (any card tally implies at least one game, so
            the baseline count is >= 1).
        """
        global_average = self._global_turn_sum / self._global_game_count
        return value_sum / count - global_average
