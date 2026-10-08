"""CardRewardReader - the card a player took (or declined) after a combat.

RAW SHAPE (extends player_history.py's): a combat point's player_stats
holds "card_choices": the card options offered, each {"card": {"id": ...},
"was_picked": bool}.

WHAT COUNTS AS A CARD REWARD: a monster, elite or boss room that offered
card choices and had at most one pick. Shops (several purchases per
visit) and events are other decisions; a reward with two or more picks is
not a one-of-N choice. Neither is read.

OPTIONS: the cards in the order shown. A skip has no picked_index.

ENCHANTMENTS are ignored: a card is its id alone.

DECK ACCOUNTING (checked on 3,000 spire_codex runs, 3,522 players): after
floor 1, every copy gained (cards_gained, or a transform's final card) is
later held in the final deck or recorded as removed / transformed away, so
nothing leaves the deck unrecorded. The only exceptions were 6 players
given a curse (CARD.DOUBT) with no cards_gained entry; the final deck
still carries it with its floor, so the timeline sees it. Floor 1 (the
run-start deck plus the Neow choice) is read from the final deck and
removal records alone.
"""

from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice
from src.data_refinement.metrics.sts2_runs.pick_choice_reader import PickChoiceReader
from src.data_refinement.metrics.sts2_runs.player_history import (
    PlayerHistory,
    PlayerPoint,
)
from src.data_refinement.metrics.sts2_runs.run_record import COMBAT_ROOM_TYPES


class CardRewardReader(PickChoiceReader):
    """Reads a combat's card reward as one pick, or a skip."""

    def _choices_at(
        self, point: PlayerPoint, history: PlayerHistory
    ) -> list[PickChoice]:
        """The reward offered at this point, as a one-element list.

        Inputs: point, history.
        Output: list[PickChoice]: empty unless point is a combat room that
            offered card choices with at most one pick.
        Side effects: as card_uuid_of.
        Exceptions: KeyError if an option lacks ["card"]["id"].

        Example:
            >>> reader._choices_at(monster_point, history)[0].picked_index
            0
        """
        # Only a combat room with a one-of-N choice
        options = point.stats.get("card_choices", [])
        if point.room_type not in COMBAT_ROOM_TYPES or not _is_one_of_n_choice(options):
            return []

        deck_before = history.timeline.raw_card_ids_before(point.floor)
        offered = [option["card"]["id"] for option in options]
        return [
            PickChoice(
                floor=point.floor,
                deck_before=self._uuids_of(deck_before),
                offered=self._uuids_of(offered),
                picked_index=_picked_index_or_skip(options),
            )
        ]


def _is_one_of_n_choice(options: list[dict]) -> bool:
    """Whether options is a one-of-N card choice: non-empty, with at
    most one option picked.

    Inputs: options (raw card_choices). Output: bool.
    Side effects: none. Exceptions: none.
    """
    return (
        bool(options) and sum(bool(option.get("was_picked")) for option in options) <= 1
    )


def _picked_index_or_skip(options: list[dict]) -> int | None:
    """The index of the picked option, or None if none was (a skip).

    Inputs: options (raw card_choices, a one-of-N choice).
    Output: int | None. Side effects: none. Exceptions: none.
    """
    for index, option in enumerate(options):
        if option.get("was_picked"):
            return index
    return None
