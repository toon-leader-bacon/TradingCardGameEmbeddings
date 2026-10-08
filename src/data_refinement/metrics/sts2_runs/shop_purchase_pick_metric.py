"""ShopPurchasePickMetric - P(card bought | deck, cards for sale, a card
was bought), modelled as a series of draft picks (shop_purchase_reader.py).

*** NOISY TARGET - READ FIRST ***
A shop purchase is decided by gold: card prices, how much the player has,
and what else they buy (relics, potions, a removal). Gold is not an input
here, which is a card-only task. The model sees the deck and the cards for
sale but not what the player could afford, so much of the signal is
unobservable. Expect accuracy far below the card reward metric; treat a
modest gain over the loss baseline as the realistic ceiling, not a defect.

Conditioned on a purchase: a visit with no purchase writes no row, and
there is no "buy nothing" option. A visit with n purchases writes n rows;
purchase k sees the stock less the k-1 cards already bought, and a deck
that holds them. Purchase order is the raw cards_gained order (assumed to
be the order bought).

Rows and columns: see pick_choice_metric.py.
"""

from pathlib import Path

from src.data_refinement.metrics.sts2_runs.pick_choice import PickKind
from src.data_refinement.metrics.sts2_runs.pick_choice_metric import PickChoiceMetric


class ShopPurchasePickMetric(PickChoiceMetric):
    """(deck, cards still for sale) -> the card bought. A noisy target:
    gold is not an input (module docstring)."""

    KIND = PickKind.SHOP_PURCHASE
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts2_runs/shop_purchase_pick.parquet")
