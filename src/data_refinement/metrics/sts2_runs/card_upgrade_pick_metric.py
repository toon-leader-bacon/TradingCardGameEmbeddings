"""CardUpgradePickMetric - P(this card upgraded | deck, a card was
upgraded): which card a player smithed at a rest site
(card_upgrade_reader.py).

The options are the distinct deck cards that still have an un-upgraded
copy. The deck context is card identity only, with no upgrade flags. A
rest site where nothing was smithed writes no row; there is no "upgrade
nothing" option (the player may have healed instead, which is not modelled).

The cleanest of the shop and rest metrics: smithing costs nothing, so no
gold noise. See card_upgrade_reader.py for how the options are counted.

Rows and columns: see pick_choice_metric.py.
"""

from pathlib import Path

from src.data_refinement.metrics.sts2_runs.pick_choice import PickKind
from src.data_refinement.metrics.sts2_runs.pick_choice_metric import PickChoiceMetric


class CardUpgradePickMetric(PickChoiceMetric):
    """(deck) -> the card upgraded at a rest site."""

    KIND = PickKind.CARD_UPGRADE
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts2_runs/card_upgrade_pick.parquet")
