"""ShopPurchaseReader - each card a player bought, as one draft-style pick
from the cards still for sale.

RAW SHAPE (checked on 3,000 spire_codex runs): at a "shop" point,
"card_choices" lists the cards still unsold at the end of the visit (so
it shrinks as the player buys), "cards_gained" lists what was bought, and
"bought_colorless" names the colorless ones among them. A shop opens with
7 card slots (the 2,183 visits with no purchase all list 7). So the stock
the player saw is the unsold cards plus the bought ones.

SERIES OF DRAFT EVENTS: a visit with n purchases becomes n PickChoices.
Purchase k offers the stock less the k-1 cards already bought and its
deck_before includes those k-1 cards. The raw order of "cards_gained" is
taken as the purchase order. The options are listed sorted by raw id,
whatever was bought, so their order cannot give the label away.

USABLE VISIT: at least one purchase, no "was_picked" flag set on the
unsold cards (a different record format, 909 visits, with 1-3 listed
cards), and unsold + bought == 7. Anything else writes no choices.

NOISE WARNING (read before trusting this metric): buying a card costs
gold, and gold is not part of the card-only input. Whether a card was
bought depends on its price, what the player could afford, and what else
they bought or skipped, none of which the model sees. Expect a noisy
target: accuracy well below the card reward's is a limit of the inputs,
not a bug. Also, the metric is conditioned on a purchase: P(card bought |
deck, stock, a card was bought). A visit with no purchase gives no row.
"""

from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice
from src.data_refinement.metrics.sts2_runs.pick_choice_reader import PickChoiceReader
from src.data_refinement.metrics.sts2_runs.player_history import (
    PlayerHistory,
    PlayerPoint,
)

_SHOP_CARD_SLOTS = 7


class ShopPurchaseReader(PickChoiceReader):
    """Reads a shop visit's purchases as a series of picks."""

    def _choices_at(
        self, point: PlayerPoint, history: PlayerHistory
    ) -> list[PickChoice]:
        """One PickChoice per card bought at this point, in purchase order.

        Inputs: point, history.
        Output: list[PickChoice]; empty unless point is a usable shop
            visit (see the module docstring).
        Side effects: as card_uuid_of. Exceptions: as read().

        Example:
            >>> len(reader._choices_at(shop_point, history))  # bought 2
            2
        """
        result: list[PickChoice] = []

        # Only a usable shop visit
        if point.room_type != "shop":
            return result
        bought = _bought_card_ids(point.stats)
        unsold = _unsold_card_ids(point.stats)
        if not _is_usable_visit(point.stats, bought, unsold):
            return result

        # Purchase k picks from the stock less the k-1 cards already bought
        arrival_deck = history.timeline.raw_card_ids_before(point.floor)
        for index, card_id in enumerate(bought):
            stock = sorted(unsold + bought[index:])
            result.append(
                PickChoice(
                    floor=point.floor,
                    deck_before=self._uuids_of(arrival_deck + bought[:index]),
                    offered=self._uuids_of(stock),
                    picked_index=stock.index(card_id),
                )
            )
        return result


def _bought_card_ids(stats: dict) -> list[str]:
    """The raw ids bought at the visit, in purchase order.

    Inputs: stats (the raw player_stats). Output: list[str], from
        "cards_gained". Side effects: none. Exceptions: KeyError if a
        gained entry lacks "id".
    """
    return [entry["id"] for entry in stats.get("cards_gained", [])]


def _unsold_card_ids(stats: dict) -> list[str]:
    """The raw ids still for sale at the end of the visit.

    Inputs: stats. Output: list[str], from "card_choices", in listed
        order. Side effects: none. Exceptions: KeyError if an entry
        lacks ["card"]["id"].
    """
    return [option["card"]["id"] for option in stats.get("card_choices", [])]


def _is_usable_visit(stats: dict, bought: list[str], unsold: list[str]) -> bool:
    """Whether the visit is a usable shop visit (module docstring).

    Inputs: stats, bought, unsold. Output: bool.
    Side effects: none. Exceptions: none.
    """
    flagged = any(option.get("was_picked") for option in stats.get("card_choices", []))
    return (
        bool(bought) and not flagged and len(bought) + len(unsold) == _SHOP_CARD_SLOTS
    )
