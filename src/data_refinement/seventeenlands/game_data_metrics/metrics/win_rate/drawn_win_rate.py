"""Win rate = fraction of games where a card was drawn that were also
won.

See src/data_refinement/seventeenlands/game_data_metrics/metric.py
for the shared Metric interface this implements, and base.py for the
shared implementation this and its two siblings (win_rate.py,
opening_hand_win_rate.py) are built on.
"""

from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.base import (
    _BinaryTriggerWinRateMetric,
)

_METRIC_NAME = "drawn_win_rate"


class DrawnWinRateMetric(_BinaryTriggerWinRateMetric):
    """Win rate triggered by CardColumnSet.drawn (card was drawn).

    See base.py's _BinaryTriggerWinRateMetric for the full
    accumulate()/finalize()/save_state()/load_state() contract — this
    subclass only fixes which metric name and which trigger column
    apply.
    """

    name: str = _METRIC_NAME
    _trigger_column = staticmethod(lambda columns: columns.drawn)
