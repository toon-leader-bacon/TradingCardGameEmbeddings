"""Average turns remaining post-cast = mean of (num_turns - turn cast),
across every cast instance — how many more turns the game lasted after
this card came down, on average. A rough "is this a game-closer played
late, or an early-game piece the game plays on for a long time after"
signal.

Precision note (settled explicitly this session, not to be
"corrected" during implementation): num_turns and a cast instance's
own turn number are on slightly different counting scales (confirmed
via real data — they don't always align exactly, by up to ~1 turn,
with no clean on-play/off-play correction rule found). This metric
computes num_turns - event.turn directly, with NO correction term.
This is an accepted approximation, relying on volume (law of large
numbers) to average out the noise — not a bug to fix later.

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

_METRIC_NAME = "average_turns_remaining_post_cast"


class AverageTurnsRemainingPostCastMetric(_CastEventAverageMetric):
    """Average turns remaining after cast, pooled across both sides and
    every cast instance.

    See base.py's _CastEventAverageMetric for the full
    accumulate()/finalize()/save_state()/load_state() contract — this
    subclass only fixes which metric name and which per-instance value
    apply.
    """

    name: str = _METRIC_NAME

    def _value(self, event: CastEvent, num_turns: int) -> float:
        """Value = num_turns - event.turn, uncorrected (see this
        module's docstring's precision note).

        Inputs:
            event: one CastEvent.
            num_turns: that row's num_turns column value.
        Output: float(num_turns - event.turn).
        Side effects: none.
        Exceptions: none expected.
        """
        return float(num_turns - event.turn)
