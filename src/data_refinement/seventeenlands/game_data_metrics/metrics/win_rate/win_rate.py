"""Win rate = fraction of games where a card was in the deck that were
also won.

See src/data_refinement/seventeenlands/game_data_metrics/metric.py
for the shared Metric interface this implements, and base.py for the
shared implementation this and its two siblings (drawn_win_rate.py,
opening_hand_win_rate.py) are built on — one implementation file per
metric under metrics/win_rate/, mirroring
src/data_refinement/card_binder/'s per-source subdirectory convention,
even though these three genuinely share logic internally.
"""

from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.base import (
    _BinaryTriggerWinRateMetric,
)

_METRIC_NAME = "win_rate"


class WinRateMetric(_BinaryTriggerWinRateMetric):
    """Win rate triggered by CardColumnSet.deck (card was in the deck).

    See base.py's _BinaryTriggerWinRateMetric for the full
    accumulate()/finalize()/save_state()/load_state() contract — this
    subclass only fixes which metric name and which trigger column
    apply.
    """

    name: str = _METRIC_NAME
    _trigger_column = staticmethod(lambda columns: columns.deck)
