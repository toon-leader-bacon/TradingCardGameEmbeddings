"""Thin wrapper over CardUpgradePickMetric
(src/data_refinement/metrics/sts2_runs/card_upgrade_pick_metric.py).

[deck cards with an un-upgraded copy, deck] -> the card upgraded at a rest
site. Smithing is free, so there is no gold noise; a rest site with no
smith writes no row, so there is no skip option.

The deck context carries card identity only, so it cannot show which copies
are already upgraded; the options do (see card_upgrade_reader.py).
"""

from src.data_refinement.metrics.sts2_runs.card_upgrade_pick_metric import (
    CardUpgradePickMetric,
)
from src.dojos.sts2_runs.sts2_run_pick_dojo import Sts2RunPickDojo


class CardUpgradePickDojo(Sts2RunPickDojo):
    """[upgradable deck cards, deck] -> the card upgraded."""

    METRIC = CardUpgradePickMetric
