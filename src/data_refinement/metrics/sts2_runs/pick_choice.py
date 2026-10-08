"""PickChoice - one "which of these options did the player take" decision,
the record every per-floor StS2 pick metric is built from.

Four decisions share this one shape (a deck on arrival, the options the
pick was made among, the option taken):
    card reward   options = the cards offered; may be declined.
    shop purchase options = the shop's cards still for sale; one choice
                  per card bought, in purchase order.
    card removal  options = the distinct cards in the deck.
    card upgrade  options = the distinct deck cards with a copy not yet
                  upgraded.
Which decision a PickChoice is, is decided by the PickKind it is stored
under on the PlayerRun (run_record.py), not by the choice itself.
"""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID


class PickKind(Enum):
    """The decisions a PickChoice can record (see the module docstring)."""

    CARD_REWARD = "card_reward"
    SHOP_PURCHASE = "shop_purchase"
    CARD_REMOVAL = "card_removal"
    CARD_UPGRADE = "card_upgrade"


@dataclass(frozen=True)
class PickChoice:
    """One decision, as the player faced it.

    floor: the floor it happened on (deck_timeline.py's numbering).
    deck_before: the deck on arrival, one card_uuid per copy (None = no
        spire_codex alias). Card identity only: no upgrade or enchantment.
        For a shop purchase it also holds the cards already bought on
        this visit.
    offered: the options, in a fixed order (None = no alias). Named for
        the card reward; for a removal or an upgrade these are the
        candidate cards.
    picked_index: the option taken, or None if the player declined (only
        a card reward can be declined).
    """

    floor: int
    deck_before: tuple[UUID | None, ...]
    offered: tuple[UUID | None, ...]
    picked_index: int | None

    def __post_init__(self) -> None:
        """Reject a picked_index outside offered.
        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError if picked_index is not None and not a
            position in offered."""
        if self.picked_index is not None and not (
            0 <= self.picked_index < len(self.offered)
        ):
            raise ValueError(
                f"picked_index {self.picked_index} is outside "
                f"{len(self.offered)} offered cards"
            )
