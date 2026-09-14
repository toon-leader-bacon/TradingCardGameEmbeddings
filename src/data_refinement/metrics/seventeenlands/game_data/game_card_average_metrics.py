"""Concrete GameCardAverageMetric (game_card_average_metric.py)
subclasses: three of round 1's six single-card metrics from
src/data_refinement/metrics/seventeenlands/game_data/BRAINSTORM.md's
"Human Review Short List" - see plans/game_data_metrics.md's Component
overview #3.

GameLengthAssociationMetric (this list's fourth win-rate-shaped
sibling) also subclasses GameCardAverageMetric, but lives in its own
file (game_length_association_metric.py) since it needs the
_extra_accumulate()/finalize() overrides that module's docstring
describes - not a plain LABEL_COLUMN/_present_card_uuids()/
_value_for_row() fixture like the three classes below.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metric import (
    GameCardAverageMetric,
)

_DEFAULT_OUTPUT_DIR = Path("data/metrics/seventeenlands/game_data")


class WinRateWhenInDeckMetric(GameCardAverageMetric):
    """P(won | card in deck_<name>) - BRAINSTORM.md's single-card
    metric "Win Rate When In Deck"."""

    LABEL_COLUMN: ClassVar[str] = "win_rate_when_in_deck"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "win_rate_when_in_deck.parquet"

    def _present_card_uuids(self, row: dict) -> list[UUID]:
        """See GameCardAverageMetric._present_card_uuids(). Cards
        present (count > 0) in deck_<name> this row."""
        return self._game_columns.present_uuids(row, self._game_columns.deck_columns)

    def _value_for_row(self, row: dict) -> float:
        """See GameCardAverageMetric._value_for_row(). 1.0 if
        row["won"] else 0.0."""
        return 1.0 if row["won"] else 0.0


class OpeningHandWinRateMetric(GameCardAverageMetric):
    """P(won | card in opening_hand_<name>) - BRAINSTORM.md's
    single-card metric "Opening Hand Win Rate"."""

    LABEL_COLUMN: ClassVar[str] = "opening_hand_win_rate"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "opening_hand_win_rate.parquet"

    def _present_card_uuids(self, row: dict) -> list[UUID]:
        """See GameCardAverageMetric._present_card_uuids(). Cards
        present (count > 0) in opening_hand_<name> this row."""
        return self._game_columns.present_uuids(
            row, self._game_columns.opening_hand_columns
        )

    def _value_for_row(self, row: dict) -> float:
        """See GameCardAverageMetric._value_for_row(). 1.0 if
        row["won"] else 0.0."""
        return 1.0 if row["won"] else 0.0


class DrawnWinRateMetric(GameCardAverageMetric):
    """P(won | card in drawn_<name>) - BRAINSTORM.md's single-card
    metric "Drawn Win Rate". Distinct from OpeningHandWinRateMetric
    since drawn_<name> counts a card seen at any point in the game,
    opening hand or not."""

    LABEL_COLUMN: ClassVar[str] = "drawn_win_rate"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "drawn_win_rate.parquet"

    def _present_card_uuids(self, row: dict) -> list[UUID]:
        """See GameCardAverageMetric._present_card_uuids(). Cards
        present (count > 0) in drawn_<name> this row."""
        return self._game_columns.present_uuids(row, self._game_columns.drawn_columns)

    def _value_for_row(self, row: dict) -> float:
        """See GameCardAverageMetric._value_for_row(). 1.0 if
        row["won"] else 0.0."""
        return 1.0 if row["won"] else 0.0
