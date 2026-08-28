"""Win rate = fraction of games where a card was in the opening hand
that were also won.

See src/data_refinement/seventeenlands/game_data_metrics/metric.py
for the shared Metric interface this implements, and base.py for the
shared implementation this and its two siblings (win_rate.py,
drawn_win_rate.py) are built on.
"""

from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.base import (
    _BinaryTriggerWinRateMetric,
)

_METRIC_NAME = "opening_hand_win_rate"


class OpeningHandWinRateMetric(_BinaryTriggerWinRateMetric):
    """Win rate triggered by CardColumnSet.opening_hand (card was kept
    in the opening hand).

    See base.py's _BinaryTriggerWinRateMetric for the full
    accumulate()/finalize()/save_state()/load_state() contract — this
    subclass only fixes which metric name and which trigger column
    apply.
    """

    name: str = _METRIC_NAME
    _trigger_column = staticmethod(lambda columns: columns.opening_hand)
