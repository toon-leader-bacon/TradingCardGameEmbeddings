"""Thin wrapper over CardRemovalPickMetric
(src/data_refinement/metrics/sts2_runs/card_removal_pick_metric.py).

[distinct deck cards, deck] -> the card removed at a shop. Removal costs
gold, which is not an input, so expect some noise (milder than the shop
purchase's). A shop visit with no removal writes no row; no skip option.

The options repeat the deck's cards, so thinning the deck context
(DECK_MOD_GROUPS, group 1) never removes the answer from the options.
"""

from src.data_refinement.metrics.sts2_runs.card_removal_pick_metric import (
    CardRemovalPickMetric,
)
from src.dojos.sts2_runs.sts2_run_pick_dojo import Sts2RunPickDojo


class CardRemovalPickDojo(Sts2RunPickDojo):
    """[distinct deck cards, deck] -> the card removed. Gold is not an
    input (module docstring)."""

    METRIC = CardRemovalPickMetric
