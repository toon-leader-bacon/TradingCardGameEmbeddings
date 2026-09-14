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

import pyarrow as pa

from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
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

    def _label_for_row(self, row: dict) -> bool:
        """See GameDeckLabelMetric._label_for_row(). row["won"]."""
        return row["won"]


class DeckGameLengthPredictionMetric(GameDeckLabelMetric):
    """Deck -> num_turns - BRAINSTORM.md's multi-card metric "Deck ->
    Game-Length Prediction"."""

    LABEL_COLUMN: ClassVar[str] = "num_turns"
    LABEL_TYPE: ClassVar[pa.DataType] = pa.int64()
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "deck_game_length_prediction.parquet"

    def _label_for_row(self, row: dict) -> int:
        """See GameDeckLabelMetric._label_for_row(). row["num_turns"]."""
        return row["num_turns"]


class DeckRankTierPredictionMetric(GameDeckLabelMetric):
    """Deck -> rank - BRAINSTORM.md's multi-card metric "Deck -> Rank-
    Tier Prediction".

    LABEL_VALUES, per game_data/BRAINSTORM.md's confirmed tier
    vocabulary: bronze, silver, gold, platinum, diamond, mythic.
    OTHER_LABEL (masked_field_metric.OTHER_LABEL, shared sentinel) is
    included as a safety net for a tier value not in this vocabulary,
    the same convention sts_gg/deck_label_metrics.py's
    CharacterPredictionMetric already established - not because one has
    been observed.
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

    def _label_for_row(self, row: dict) -> str:
        """See GameDeckLabelMetric._label_for_row(). row["rank"] if it
        is a member of self.LABEL_VALUES, else OTHER_LABEL."""
        rank = row["rank"]
        if rank not in self.LABEL_VALUES:
            return OTHER_LABEL
        return rank
