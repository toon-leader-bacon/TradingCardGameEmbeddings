"""PLACEHOLDER — not yet designed for implementation. See
average_turn_cast.py's "Open design question" note, which applies here
identically.

Average turns held in hand = for games where a card was drawn/opened,
the mean number of turns between it first appearing in an
eot_*_cards_in_hand column and it leaving hand (cast or discarded, or
game end if neither happened first). A "how long do you sit on this
card before acting on it" signal.

Raw columns: eot_user_cards_in_hand / eot_oppo_cards_in_hand across all
turns (whichever side holds the card in a given game), cross-referenced
against the cast/discard columns average_turn_cast.py and
discard_rate.py also need, to find the turn it left hand.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "average_turns_held_in_hand"


class AverageTurnsHeldInHandMetric:
    """Placeholder — see this module's docstring."""

    name: str = _METRIC_NAME

    def __init__(self, expansion: str, format_code: str) -> None:
        """
        Inputs:
            expansion: 17lands expansion code, stamped onto results.
            format_code: 17lands format code, stamped onto results.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        raise NotImplementedError

    def accumulate(self, chunk: pd.DataFrame, resolved: pd.Series) -> None:
        """Placeholder — see this module's docstring."""
        raise NotImplementedError

    def finalize(self) -> dict[UUID, MetricResult]:
        """Placeholder — see this module's docstring."""
        raise NotImplementedError

    def save_state(self) -> dict:
        """Placeholder — see this module's docstring."""
        raise NotImplementedError

    def load_state(self, state: dict) -> None:
        """Placeholder — see this module's docstring."""
        raise NotImplementedError
