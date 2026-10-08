"""Thin wrapper over ShopPurchasePickMetric
(src/data_refinement/metrics/sts2_runs/shop_purchase_pick_metric.py).

*** NOISY TARGET ***
[cards for sale, deck] -> the card bought. The player's gold, the prices and
the rest of their basket decide a purchase, and none of them is an input to
this card-only task, so a loss that sits close to the baseline is the
expected outcome, not a training failure. Judge it by a small, steady edge
over the baseline (BaselineLoss), not by accuracy.

A visit with no purchase writes no row, so there is no skip option
(CAN_SKIP is False). A visit with several purchases writes one row per
purchase, each seeing the stock less the earlier ones.
"""

from src.data_refinement.metrics.sts2_runs.shop_purchase_pick_metric import (
    ShopPurchasePickMetric,
)
from src.dojos.sts2_runs.sts2_run_pick_dojo import Sts2RunPickDojo


class ShopPurchasePickDojo(Sts2RunPickDojo):
    """[cards for sale, deck] -> the card bought. A noisy target: gold is
    not an input (module docstring)."""

    METRIC = ShopPurchasePickMetric
