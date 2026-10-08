"""Thin wrapper over CardRewardPickMetric
(src/data_refinement/metrics/sts2_runs/card_reward_pick_metric.py).

[cards offered, deck so far] in, the card taken out, or a learned "none of
them" option when the player skipped the reward (CAN_SKIP).

SKIP LIMIT: the skip option is one learned vector scored against the deck
context, so it cannot see how good the offered cards are; weak skip
accuracy is a limit of that design, not a bug.
"""

from src.data_refinement.metrics.sts2_runs.card_reward_pick_metric import (
    CardRewardPickMetric,
)
from src.dojos.sts2_runs.sts2_run_pick_dojo import Sts2RunPickDojo


class CardRewardPickDojo(Sts2RunPickDojo):
    """[cards offered, deck so far] -> the card taken, or none
    (CardRewardPickMetric)."""

    METRIC = CardRewardPickMetric
    CAN_SKIP = True
