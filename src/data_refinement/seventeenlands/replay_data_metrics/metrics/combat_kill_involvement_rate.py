"""PLACEHOLDER — not yet designed for implementation. See
average_turn_cast.py's "Open design question" note, which applies here
identically.

Combat kill involvement rate = of turns a card attacked or blocked,
the fraction where it appears in that same turn's
user_creatures_killed_combat or oppo_creatures_killed_combat column
(on either side — being the creature that dealt a lethal blow, or the
one that was killed, are two different questions this single metric
does NOT yet distinguish; splitting into separate "kills in combat"
vs. "dies in combat" jobs is a likely refinement once this is actually
designed).

Raw columns: creatures_attacked, creatures_blocked, creatures_blocking,
user_creatures_killed_combat, oppo_creatures_killed_combat — all
confirmed present with per-card Arena IDs in real data.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "combat_kill_involvement_rate"


class CombatKillInvolvementRateMetric:
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
