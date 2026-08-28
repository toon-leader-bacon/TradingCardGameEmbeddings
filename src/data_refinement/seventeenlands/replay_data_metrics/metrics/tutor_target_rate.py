"""PLACEHOLDER — not yet designed for implementation. See
average_turn_cast.py's "Open design question" note, which applies here
identically.

Tutor target rate = how often a card is the one fetched by a tutor
effect, across every observed cards_tutored occurrence in the file
(confirmed via real data: cards_tutored cells hold Arena IDs, e.g.
'79741', '102732', '34742' — the card that was found, not the tutor
spell itself). Unlike this file's other rate-style metrics, there's no
natural "games this card was eligible" denominator available from
replay_data alone (no deck-membership signal here — that lives in
game_data) — candidate denominators (total tutor events in the file;
games this card appeared in by any means) need to be decided before
this is implementable, not just its column resolution mechanism.

Raw columns: cards_tutored, both user_turn_N and oppo_turn_N variants,
across all turns.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "tutor_target_rate"


class TutorTargetRateMetric:
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
