"""CardRemovalPickMetric - P(this card removed | deck, a card was removed):
which card a player paid to remove at a shop (card_removal_reader.py).

The options are the distinct cards of the deck, so the label is a card,
not one of several identical copies. A shop visit with no removal writes no
row; there is no "remove nothing" option.

GOLD: removal costs gold, which is not an input; the player may remove a
card only when they can afford it. This is a milder source of noise than
for the shop purchase (the metric is conditioned on a removal having been
paid for), but expect a noisier target than the card upgrade.

Rows and columns: see pick_choice_metric.py.
"""

from pathlib import Path

from src.data_refinement.metrics.sts2_runs.pick_choice import PickKind
from src.data_refinement.metrics.sts2_runs.pick_choice_metric import PickChoiceMetric


class CardRemovalPickMetric(PickChoiceMetric):
    """(deck) -> the card removed at a shop. Gold is not an input (module
    docstring)."""

    KIND = PickKind.CARD_REMOVAL
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts2_runs/card_removal_pick.parquet")
