"""PLACEHOLDER — not yet designed for implementation. See
average_turn_cast.py's "Open design question" note, which applies here
identically.

Discard rate = of games where a card was drawn/opened, the fraction of
times it ends up in a cards_discarded column instead of being cast. A
"how often does this get thrown away rather than played" signal —
shares the same "games this card was seen at all" denominator question
tutor_target_rate.py flags, since replay_data has no deck-membership
signal of its own.

Raw columns: cards_discarded, both user_turn_N and oppo_turn_N
variants, across all turns; a denominator drawn from wherever this
container ends up tracking "card was in a hand this game" (opening_hand,
candidate_hand_*, or eot_*_cards_in_hand — not yet decided which).
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "discard_rate"


class DiscardRateMetric:
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
