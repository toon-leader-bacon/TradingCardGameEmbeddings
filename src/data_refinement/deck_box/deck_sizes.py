"""How many cards a full deck holds: the size check deck stages share
(limitless_play/, mtg_crawls/)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeckSizes:
    """How many cards a full deck holds, bounds inclusive.

    max_cards None means no upper bound (Lorcana, Star Wars: Unlimited).
    """

    min_cards: int
    max_cards: int | None

    def __post_init__(self) -> None:
        """Reject bounds no deck fits.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError if min_cards < 1 or max_cards < min_cards.
        """
        if self.min_cards < 1:
            raise ValueError(f"min_cards must be at least 1, not {self.min_cards}")
        if self.max_cards is not None and self.max_cards < self.min_cards:
            raise ValueError(
                f"max_cards {self.max_cards} is below min_cards {self.min_cards}"
            )

    def fits(self, card_count: int) -> bool:
        """Whether card_count is a full deck's size.

        Inputs: card_count. Output: bool. Side effects: none.
        Exceptions: none.

        Example:
            >>> DeckSizes(50, 55).fits(54)
            True
            >>> DeckSizes(60, None).fits(80)
            True
        """
        if card_count < self.min_cards:
            return False
        return self.max_cards is None or card_count <= self.max_cards
