"""A player's deck over a run, rebuilt from the cards' own floors.

The raw run stores no deck snapshots: only the final deck (each copy with
the floor it entered) and, per floor, the copies that left it (removed at a
shop, transformed at an event). A copy was in the deck on arrival at floor
F exactly when it entered before F and had not yet left by F. So the deck
at any floor is a filter over every copy the player ever held, which this
module holds as TimedCards.

Card ids here are the raw "CARD.<NAME>" strings: this module neither
reads the raw run dict nor looks cards up (the reader that builds a
DeckTimeline does both; see card_reward_reader.py).

FLOORS count map points across all acts from 1, the numbering
"floor_added_to_deck" uses (the run-start deck is floor 1).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimedCard:
    """One copy of a card the player held at some point in the run.

    raw_card_id: the raw "CARD.<NAME>" id.
    floor_added: the floor the copy entered the deck (1 = run start).
    floor_left: the floor a shop removal or event transform took it out
        of the deck, or None if it was still held at run end.
    """

    raw_card_id: str
    floor_added: int
    floor_left: int | None


@dataclass(frozen=True)
class DeckTimeline:
    """Every copy a player ever held (see the module docstring)."""

    cards: tuple[TimedCard, ...]

    def raw_card_ids_before(self, floor: int) -> list[str]:
        """The deck on arrival at floor, one raw id per copy.

        Inputs: floor (int, >= 1).
        Output: list[str]: ids of copies added before floor and not left
            before it, in the order the timeline holds them.
        Side effects: none.
        Exceptions: ValueError if floor < 1.

        Example:
            >>> timeline.raw_card_ids_before(2)  # the run-start deck
            ['CARD.STRIKE_REGENT', 'CARD.STRIKE_REGENT', 'CARD.DEFEND_REGENT']
        """
        result: list[str] = []

        # Validate inputs
        if floor < 1:
            raise ValueError(f"floor must be >= 1, got {floor}")

        # Keep each copy that had entered, and not left, by this floor
        for card in self.cards:
            if card.floor_added >= floor:
                continue
            if card.floor_left is not None and card.floor_left < floor:
                continue
            result.append(card.raw_card_id)
        return result
