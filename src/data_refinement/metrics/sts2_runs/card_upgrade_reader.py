"""CardUpgradeReader - the card a player upgraded at a rest-site smith.

RAW SHAPE: at a "rest_site" point, "rest_site_choices" lists the actions
taken ("SMITH" among them) and "upgraded_cards" the raw ids upgraded. A
smith upgrades exactly one card (10,497 of 10,497 plain-smith visits);
visits with two or more upgraded cards come from relics or other effects
and are not read. Upgrades at events and shops are other decisions and are
not read.

OPTIONS: the distinct deck cards that still have a copy not yet upgraded,
sorted by uuid. A card still has an un-upgraded copy when the deck on
arrival holds more copies of it than PlayerHistory counts upgrades of it
so far, over every room type. This is approximate: an upgraded copy
that was later removed or transformed away still counts as an upgrade, so
a card can be wrongly left out, and then the visit writes no row (about
2% of smith visits on 3,000 real runs, with unaliased cards).

The model sees the deck as card identity only (no upgrade flags), so the
context does not say which copies are already upgraded; the options do.

GOLD: none; smithing is free. The choice is steered by the rest site's
other options (heal) and the player's HP, which are not inputs, but the
target is a clean one.
"""

from collections import Counter
from typing import Mapping

from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice
from src.data_refinement.metrics.sts2_runs.pick_choice_reader import PickChoiceReader
from src.data_refinement.metrics.sts2_runs.player_history import (
    PlayerHistory,
    PlayerPoint,
)


class CardUpgradeReader(PickChoiceReader):
    """Reads a smith's upgrade as one pick among the upgradable cards."""

    def _choices_at(
        self, point: PlayerPoint, history: PlayerHistory
    ) -> list[PickChoice]:
        """The upgrade made at this point, as a one-element list.

        Inputs: point, history.
        Output: list[PickChoice]: empty unless point is a rest site where
            "SMITH" was chosen and exactly one card was upgraded, that
            card has an alias, and it is among the options.
        Side effects: as card_uuid_of. Exceptions: as read().

        Example:
            >>> reader._choices_at(rest_point, history)[0].floor
            9
        """
        # Only a smith that upgraded exactly one card
        upgraded = point.stats.get("upgraded_cards", [])
        smithed = "SMITH" in point.stats.get("rest_site_choices", [])
        if point.room_type != "rest_site" or not smithed or len(upgraded) != 1:
            return []

        # Options: cards with a copy not yet upgraded on arrival
        arrival_deck = history.timeline.raw_card_ids_before(point.floor)
        already_upgraded = history.upgrade_counts_before(point.floor)
        options = self._distinct_uuids_of(
            _with_unupgraded_copy(arrival_deck, already_upgraded)
        )
        return self._pick_among_distinct(point, arrival_deck, options, upgraded[0])


def _with_unupgraded_copy(
    deck: list[str], upgrade_counts: Mapping[str, int]
) -> list[str]:
    """The ids in deck that hold more copies than they have upgrades, in
    first-appearance order, one entry per distinct id.

    Inputs: deck (raw ids, one per copy), upgrade_counts (raw id ->
        upgrades so far). Output: list[str].
    Side effects: none. Exceptions: none.

    Example:
        >>> _with_unupgraded_copy(["A", "A", "B"], {"A": 2})
        ['B']
    """
    copies = Counter(deck)
    return [
        card_id
        for card_id, count in copies.items()
        if count > upgrade_counts.get(card_id, 0)
    ]
