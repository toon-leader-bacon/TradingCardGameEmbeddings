"""Average turn cast = mean turn number a card is cast on, across every
cast instance (not just the first per game — see base.py's docstring
for why every instance is its own observation).

See base.py for the shared implementation this is a thin subclass of,
and cast_event_scanner.py for how cast instances get found in the
first place.
"""

from src.data_refinement.seventeenlands.replay_data_metrics.metrics.cast_event_average.base import (
    _CastEventAverageMetric,
)
from src.data_refinement.seventeenlands.replay_data_metrics.cast_event_scanner import (
    CastEvent,
)

_METRIC_NAME = "average_turn_cast"


class AverageTurnCastMetric(_CastEventAverageMetric):
    """Average turn cast, pooled across both sides and every cast instance.

    See base.py's _CastEventAverageMetric for the full
    accumulate()/finalize()/save_state()/load_state() contract — this
    subclass only fixes which metric name and which per-instance value
    apply.
    """

    name: str = _METRIC_NAME

    def _value(self, event: CastEvent, num_turns: int) -> float:
        """Value = event.turn (the turn this instance was cast on).

        Inputs:
            event: one CastEvent.
            num_turns: unused by this subclass.
        Output: float(event.turn).
        Side effects: none.
        Exceptions: none expected.
        """
        return float(event.turn)
