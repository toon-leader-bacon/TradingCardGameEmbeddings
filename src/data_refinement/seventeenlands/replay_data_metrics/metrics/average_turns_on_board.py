"""PLACEHOLDER — not yet designed for implementation. See
average_turn_cast.py's "Open design question" note, which applies here
identically.

Average turns on board = for games where a permanent (creature or
non-creature) was cast, the mean number of turns between it entering
play and leaving eot_*_creatures_in_play / eot_*_non_creatures_in_play
(removed, or game end if it never leaves). A "how long does this
stick around once it resolves" signal — irrelevant/always-zero for
instants and sorceries, so this job is naturally permanents-only
(cards never seen in an eot_*_creatures_in_play or
eot_*_non_creatures_in_play column contribute no observations, the
same "leave zero-observation cards untouched" pattern as
_binary_trigger_rate_family.py).

Raw columns: eot_user_creatures_in_play / eot_oppo_creatures_in_play /
eot_user_non_creatures_in_play / eot_oppo_non_creatures_in_play across
all turns.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "average_turns_on_board"


class AverageTurnsOnBoardMetric:
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
