"""CardRemovalReader - the card a player paid to remove at a shop.

RAW SHAPE: at a "shop" point, "cards_removed" lists the copies removed
([{"id", "floor_added_to_deck"}]); the shop's removal service takes at most
one (4,088 of 4,088 visits with a removal had exactly one). Removals at
events, rests and combats are other decisions (forced, or one option among
an event's choices) and are not read.

OPTIONS: the distinct cards in the deck on arrival, sorted by uuid. Three
Strikes are one option, so the label never splits probability among
identical copies; the dojo's deck context still shows how many copies
there are.

A removal whose card has no alias writes no choice (it cannot be a label).
Unaliased deck cards are left out of the options.

GOLD: removal costs gold, which rises with each use, and gold is not an
input. The decision is also steered by what the player can afford, so
expect some noise, though far less than the shop purchase has.
"""

from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice
from src.data_refinement.metrics.sts2_runs.pick_choice_reader import PickChoiceReader
from src.data_refinement.metrics.sts2_runs.player_history import (
    PlayerHistory,
    PlayerPoint,
)


class CardRemovalReader(PickChoiceReader):
    """Reads a shop's card removal as one pick among the deck's cards."""

    def _choices_at(
        self, point: PlayerPoint, history: PlayerHistory
    ) -> list[PickChoice]:
        """The removal made at this point, as a one-element list.

        Inputs: point, history.
        Output: list[PickChoice]: empty unless point is a shop with
            exactly one removed card that has an alias and is among the
            deck's options.
        Side effects: as card_uuid_of. Exceptions: as read().

        Example:
            >>> reader._choices_at(shop_point, history)[0].picked_index
            0
        """
        # Only a shop with exactly one removal
        removed = point.stats.get("cards_removed", [])
        if point.room_type != "shop" or len(removed) != 1:
            return []

        # The options are the distinct cards of the deck on arrival
        arrival_deck = history.timeline.raw_card_ids_before(point.floor)
        options = self._distinct_uuids_of(arrival_deck)
        return self._pick_among_distinct(point, arrival_deck, options, removed[0]["id"])
