"""PickChoiceReader - the base of the readers that turn one raw player
into PickChoices (pick_choice.py).

Template Method: read() walks the player's points; each subclass says what
a single point contributes (nothing, one choice, or several) in
_choices_at(). The card-id -> uuid mapping, and the rule that an unaliased
card is None, live here once.
"""

from abc import ABC, abstractmethod
from typing import Callable
from uuid import UUID

from src.data_refinement.metrics.sts2_runs.pick_choice import PickChoice
from src.data_refinement.metrics.sts2_runs.player_history import (
    PlayerHistory,
    PlayerPoint,
)


class PickChoiceReader(ABC):
    """Reads one kind of decision off a PlayerHistory."""

    def __init__(self, card_uuid_of: Callable[[str], UUID | None]) -> None:
        """
        Inputs:
            card_uuid_of: raw "CARD.<NAME>" id -> the card's nocab_uuid,
                or None when the id has no alias (the parser's cached
                lookup, so a miss is logged once).
        Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        self._card_uuid_of = card_uuid_of

    def read(self, history: PlayerHistory) -> tuple[PickChoice, ...]:
        """Every decision of this kind the player made, in floor order.

        Inputs: history (the player's points and timeline).
        Output: tuple[PickChoice, ...]; empty if none qualifies.
        Side effects: whatever card_uuid_of does (caches, logs a miss).
        Exceptions: KeyError / TypeError if a point lacks a field the
            subclass reads.

        Example:
            >>> CardRemovalReader(uuid_of).read(history)[0].floor
            23
        """
        result: list[PickChoice] = []

        # What each point contributes, in floor order
        for point in history.points:
            result.extend(self._choices_at(point, history))
        return tuple(result)

    @abstractmethod
    def _choices_at(
        self, point: PlayerPoint, history: PlayerHistory
    ) -> list[PickChoice]:
        """The decisions of this kind made at one point.

        Inputs: point, history (for the deck on arrival and upgrades).
        Output: list[PickChoice]; empty when the point is not this kind
            of decision, or does not meet the subclass's usability rules.
        Side effects: as card_uuid_of. Exceptions: as read().
        """

    def _uuids_of(self, raw_card_ids: list[str]) -> tuple[UUID | None, ...]:
        """Map raw ids through card_uuid_of, keeping order and None.

        Inputs: raw_card_ids. Output: tuple, one entry per id.
        Side effects: as card_uuid_of. Exceptions: none.
        """
        return tuple(self._card_uuid_of(raw_id) for raw_id in raw_card_ids)

    def _distinct_uuids_of(self, raw_card_ids: list[str]) -> tuple[UUID, ...]:
        """The distinct aliased uuids of raw_card_ids, sorted (copies of
        one card count once; unaliased ids are left out). Sorted so the
        options' order says nothing about the deck's history.

        Inputs: raw_card_ids. Output: tuple[UUID, ...].
        Side effects: as card_uuid_of. Exceptions: none.

        Example:
            >>> reader._distinct_uuids_of(["CARD.A", "CARD.A", "CARD.B"])
            (UUID('...a'), UUID('...b'))
        """
        uuids = (self._card_uuid_of(raw_id) for raw_id in raw_card_ids)
        return tuple(sorted({uuid for uuid in uuids if uuid is not None}))

    def _pick_among_distinct(
        self,
        point: PlayerPoint,
        arrival_deck: list[str],
        options: tuple[UUID, ...],
        picked_raw_id: str,
    ) -> list[PickChoice]:
        """The one-element result of a removal or an upgrade: a pick of
        one card among the options.

        Inputs: point, arrival_deck (raw ids on arrival), options (from
            _distinct_uuids_of), picked_raw_id (the card removed or
            upgraded).
        Output: [PickChoice]; [] if the picked card has no alias or is not
            among options (the timeline can disagree with the raw event).
        Side effects: as card_uuid_of. Exceptions: none.
        """
        picked_uuid = self._card_uuid_of(picked_raw_id)
        if picked_uuid is None or picked_uuid not in options:
            return []
        return [
            PickChoice(
                floor=point.floor,
                deck_before=self._uuids_of(arrival_deck),
                offered=options,
                picked_index=options.index(picked_uuid),
            )
        ]
