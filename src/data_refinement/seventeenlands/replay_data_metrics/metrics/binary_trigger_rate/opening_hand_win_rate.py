"""Opening hand win rate = fraction of games where a card was in the
kept opening hand that were also won.

See base.py for the shared implementation this is a thin subclass of,
and arena_id_cache.py for how opening_hand cells get resolved into
ResolvedHands.opening_hand before this metric ever sees a chunk.
"""

import pandas as pd

from src.data_refinement.seventeenlands.replay_data_metrics.metrics.binary_trigger_rate.base import (
    _BinaryTriggerRateMetric,
)

_METRIC_NAME = "opening_hand_win_rate"


class OpeningHandWinRateMetric(_BinaryTriggerRateMetric):
    """Win rate triggered by presence in the kept opening hand.

    See base.py's _BinaryTriggerRateMetric for the full
    accumulate()/finalize()/save_state()/load_state() contract — this
    subclass only fixes which metric name, which ResolvedHands field,
    and which outcome apply.
    """

    name: str = _METRIC_NAME
    _hand_field = "opening_hand"

    def _outcome(self, chunk: pd.DataFrame, resolved: pd.Series) -> pd.Series:
        """Outcome = chunk["won"], the raw per-game result column.

        Inputs:
            chunk: the raw chunk, as passed to accumulate() — must
                include a "won" column.
            resolved: unused by this subclass (the outcome here is a
                raw chunk column, not a resolved field).
        Output: chunk["won"].
        Side effects: none.
        Exceptions: none expected.
        """
        return chunk["won"]
