"""PLACEHOLDER — not yet designed for implementation. See
average_turn_cast.py's "Open design question" note, which applies here
identically.

Attack blocked rate = of turns a card attacked, the fraction where it
appears in that same turn's creatures_blocked column rather than
creatures_unblocked — a "how often does this actually get through"
signal, at attack-instance granularity (one card attacking on N
different turns across N different games all count separately, unlike
this container's game-level rate jobs).

Raw columns: creatures_attacked, creatures_blocked, creatures_unblocked
— all three confirmed present with per-card Arena IDs in real data
(user_turn_3_creatures_attacked: '104936', '104970|104941', ...).
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "attack_blocked_rate"


class AttackBlockedRateMetric:
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
