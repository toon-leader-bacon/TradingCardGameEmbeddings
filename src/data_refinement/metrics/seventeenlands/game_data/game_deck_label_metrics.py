"""Concrete GameDeckLabelMetric (game_deck_label_metric.py) subclasses:
three of round 1's five multi-card metrics from
src/data_refinement/metrics/seventeenlands/game_data/BRAINSTORM.md's
"Human Review Short List" - see plans/game_data_metrics.md's Component
overview #8.

OnPlayWinRateSensitivityByDeckMetric (this list's fourth deck-input
sibling) does NOT subclass GameDeckLabelMetric - it's accumulation, not
streaming, since its label needs cross-row aggregation across every
game sharing an identical deck (see
on_play_win_rate_sensitivity_by_deck_metric.py's own module docstring).
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metric import (
    GameDeckLabelMetric,
)

_DEFAULT_OUTPUT_DIR = Path("data/metrics/seventeenlands/game_data")


class DeckWinPredictionMetric(GameDeckLabelMetric):
    """Deck -> won - BRAINSTORM.md's multi-card metric "Deck Composition
    -> Win Prediction"."""

    LABEL_COLUMN: ClassVar[str] = "won"
    LABEL_TYPE: ClassVar[pa.DataType] = pa.bool_()
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "deck_win_prediction.parquet"

    def _labels(self, chunk: GameDataChunk) -> npt.NDArray[np.bool_]:
        """See GameDeckLabelMetric._labels(). chunk.won."""
        return chunk.won


class DeckGameLengthPredictionMetric(GameDeckLabelMetric):
    """Deck -> num_turns - BRAINSTORM.md's multi-card metric "Deck ->
    Game-Length Prediction"."""

    LABEL_COLUMN: ClassVar[str] = "num_turns"
    LABEL_TYPE: ClassVar[pa.DataType] = pa.int64()
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "deck_game_length_prediction.parquet"

    def _labels(self, chunk: GameDataChunk) -> npt.NDArray[np.int64]:
        """See GameDeckLabelMetric._labels(). chunk.num_turns as int64."""
        return chunk.num_turns.astype(np.int64)


class DeckRankTierPredictionMetric(GameDeckLabelMetric):
    """Deck -> rank - BRAINSTORM.md's multi-card metric "Deck -> Rank-
    Tier Prediction".

    LABEL_VALUES, per game_data/BRAINSTORM.md's confirmed tier
    vocabulary: bronze, silver, gold, platinum, diamond, mythic.
    OTHER_LABEL (masked_field_metric.OTHER_LABEL, shared sentinel) is
    the label for any rank outside that vocabulary - in practice the
    empty rank of every unranked (Trad, Sealed) event, which the row
    implementation read as a null and also wrote as OTHER_LABEL.
    """

    LABEL_COLUMN: ClassVar[str] = "rank"
    LABEL_TYPE: ClassVar[pa.DataType] = pa.string()
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "deck_rank_tier_prediction.parquet"
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "bronze",
        "silver",
        "gold",
        "platinum",
        "diamond",
        "mythic",
        OTHER_LABEL,
    )

    def _labels(self, chunk: GameDataChunk) -> npt.NDArray[np.object_]:
        """See GameDeckLabelMetric._labels(). chunk.rank where it is a
        member of self.LABEL_VALUES, else OTHER_LABEL."""
        known = np.isin(chunk.rank, self.LABEL_VALUES)
        return np.where(known, chunk.rank, OTHER_LABEL)
