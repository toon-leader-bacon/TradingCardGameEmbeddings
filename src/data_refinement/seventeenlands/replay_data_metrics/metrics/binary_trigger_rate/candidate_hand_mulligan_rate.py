"""Candidate hand mulligan rate = fraction of games where a card was in
the FIRST hand seen (pre-mulligan-decision) that were mulliganed away.

See base.py for the shared implementation this is a thin subclass of,
and arena_id_cache.py for how candidate_hand_1 cells get resolved into
ResolvedHands.candidate_hand_1 (and mulliganed derived onto the same
object) before this metric ever sees a chunk. Deliberately scoped to
candidate_hand_1 only, not every candidate_hand_1..7 column —
candidate_hand_1 is always the first 7 cards drawn, the one hand every
game has in common regardless of how many mulligans followed, so it's
the only candidate_hand column with a consistent per-card meaning
across games (candidate_hand_2's existence already depends on whether a
mulligan happened at all).

A high value here doesn't mean the card itself is "bad" in any general
sense — it's a proxy for "hands containing this card, specifically as
part of the FIRST 7 cards seen, tended to get mulliganed" (e.g. a
notoriously slow/situational card, or just noisy correlation with
whatever else that opening 7 happened to contain). Treat as a rough
signal, not a verdict on the card.
"""

import pandas as pd

from src.data_refinement.seventeenlands.replay_data_metrics.metrics.binary_trigger_rate.base import (
    _BinaryTriggerRateMetric,
)

_METRIC_NAME = "candidate_hand_mulligan_rate"


class CandidateHandMulliganRateMetric(_BinaryTriggerRateMetric):
    """Mulligan rate triggered by presence in the first hand seen.

    See base.py's _BinaryTriggerRateMetric for the full
    accumulate()/finalize()/save_state()/load_state() contract — this
    subclass only fixes which metric name, which ResolvedHands field,
    and which outcome apply.
    """

    name: str = _METRIC_NAME
    _hand_field = "candidate_hand_1"

    def _outcome(self, chunk: pd.DataFrame, resolved: pd.Series) -> pd.Series:
        """Outcome = resolved[row].mulliganed for each row.

        Inputs:
            chunk: unused by this subclass (the outcome here is a
                derived ResolvedHands field, not a raw chunk column).
            resolved: the ResolvedHands Series, as passed to
                accumulate().
        Output: resolved.map(lambda r: r.mulliganed).
        Side effects: none.
        Exceptions: none expected.
        """
        return resolved.map(lambda r: r.mulliganed)
