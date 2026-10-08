"""CardRewardPickMetric - BRAINSTORM.md #15 (Draft Pick Prediction): given
the deck so far and the cards a reward offered, which one was taken, or
none.

One output row per card-reward choice a scored player faced (about 17 rows
per run, so the scan writes on the order of 10^7 rows). The only pick
metric whose picked_uuid can be NULL: NULL means the player skipped.

Rows and columns: see pick_choice_metric.py.
"""

from pathlib import Path

from src.data_refinement.metrics.sts2_runs.pick_choice import PickKind
from src.data_refinement.metrics.sts2_runs.pick_choice_metric import PickChoiceMetric


class CardRewardPickMetric(PickChoiceMetric):
    """(deck so far, cards offered) -> the card taken, or none."""

    KIND = PickKind.CARD_REWARD
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts2_runs/card_reward_pick.parquet")
