"""Concrete GameCardAverageMetric (game_card_average_metric.py)
subclasses: the three per-zone win rates.

All three average the same per-game value (won as 1.0/0.0) and differ
only in which zone counts as "present", so they share
GameCardWinRateMetric's _values() and each sets just ZONE, LABEL_COLUMN
and DEFAULT_OUTPUT_PATH.

GameLengthAssociationMetric, the fourth GameCardAverageMetric, lives in
its own file: it averages num_turns and overrides _extra_accumulate()
and _label() to subtract a format-wide baseline.
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

_DEFAULT_OUTPUT_DIR = Path("data/metrics/seventeenlands/game_data")


class GameCardWinRateMetric(GameCardAverageMetric):
    """P(won | card present in ZONE): the shared value step of the
    three win-rate metrics below. Abstract (no ZONE or LABEL_COLUMN)."""

    def _values(self, chunk: GameDataChunk) -> npt.NDArray[np.float64]:
        """See GameCardAverageMetric._values(): won as 1.0 / 0.0.

        Inputs: chunk. Output: shape (len(chunk),).
        Side effects: none. Exceptions: none.
        """
        return chunk.won.astype(np.float64)


class WinRateWhenInDeckMetric(GameCardWinRateMetric):
    """P(won | card in deck_<name>) - BRAINSTORM.md's "Win Rate When In
    Deck"."""

    LABEL_COLUMN: ClassVar[str] = "win_rate_when_in_deck"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "win_rate_when_in_deck.parquet"
    ZONE: ClassVar[GameZone] = GameZone.DECK


class OpeningHandWinRateMetric(GameCardWinRateMetric):
    """P(won | card in opening_hand_<name>) - BRAINSTORM.md's "Opening
    Hand Win Rate"."""

    LABEL_COLUMN: ClassVar[str] = "opening_hand_win_rate"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "opening_hand_win_rate.parquet"
    ZONE: ClassVar[GameZone] = GameZone.OPENING_HAND


class DrawnWinRateMetric(GameCardWinRateMetric):
    """P(won | card in drawn_<name>) - BRAINSTORM.md's "Drawn Win Rate".
    Distinct from OpeningHandWinRateMetric: drawn_<name> counts a card
    seen at any point in the game, opening hand or not."""

    LABEL_COLUMN: ClassVar[str] = "drawn_win_rate"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "drawn_win_rate.parquet"
    ZONE: ClassVar[GameZone] = GameZone.DRAWN
