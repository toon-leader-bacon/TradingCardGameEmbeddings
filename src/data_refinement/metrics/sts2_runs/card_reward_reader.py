"""Raw StS2 player -> that player's card-reward choices (CardRewardChoice,
run_record.py): the part of the raw run schema the per-floor metrics read.

RAW SHAPE (extends run_parser.py's): each map point's "rooms" holds the
room_type; its "player_stats" holds one entry per player, matched by
"player_id", with:
    - "card_choices": the card options offered, each {"card": {"id": ...},
      "was_picked": bool};
    - "cards_removed": [{"id", "floor_added_to_deck"}] copies removed;
    - "cards_transformed": [{"original_card": {...}, "final_card": {...}}]
      (the original copy leaves the deck; the final card is a new copy
      that also appears in the final deck or a later removal).
A player's "deck" entries carry "floor_added_to_deck" (the floor numbering
in deck_timeline.py).

WHAT COUNTS AS A CARD REWARD: a monster, elite or boss room that offered
card choices and had at most one pick. Shops (several purchases per
visit) and events are other decisions; a reward with two or more picks is
not a one-of-N choice. Neither is read.

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

from dataclasses import dataclass
from typing import Callable
from uuid import UUID

from src.data_refinement.metrics.sts2_runs.deck_timeline import (
    DeckTimeline,
    TimedCard,
)
from src.data_refinement.metrics.sts2_runs.run_record import (
    COMBAT_ROOM_TYPES,
    CardRewardChoice,
)


@dataclass(frozen=True)
class _PlayerPoint:
    """One map point as one player saw it.

    floor: the point's 1-based position across all acts.
    room_type: the raw room_type of its first room.
    stats: the raw player_stats entry for this player.
    """

    floor: int
    room_type: str
    stats: dict


class CardRewardReader:
    """Reads a raw player's card-reward choices (see the module docstring)."""

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

    def read(
        self, player: dict, map_points: list[dict]
    ) -> tuple[CardRewardChoice, ...]:
        """Every card-reward choice this player faced, in floor order.

        Inputs:
            player: one raw player dict ("id", "deck").
            map_points: every act's map points, flattened, in order.
        Output: tuple[CardRewardChoice, ...]; empty if the player saw no
            reward that qualifies.
        Side effects: whatever card_uuid_of does (caches, logs a miss).
        Exceptions: KeyError / TypeError if a map point or player lacks a
            field the module docstring names.

        Example:
            >>> reader.read(raw_player, flat_map_points)[0].picked_index
            0
        """
        result: list[CardRewardChoice] = []
        points = self._player_points(player["id"], map_points)
        timeline = self._build_deck_timeline(player, points)

        # One choice per qualifying reward room
        for point in points:
            if point.room_type not in COMBAT_ROOM_TYPES:
                continue
            options = point.stats.get("card_choices", [])
            if not _is_one_of_n_choice(options):
                continue
            picked_index = _picked_index_or_skip(options)
            result.append(self._choice_at(point.floor, options, picked_index, timeline))
        return tuple(result)

    def _player_points(
        self, player_id: str, map_points: list[dict]
    ) -> list[_PlayerPoint]:
        """Each map point paired with this player's own player_stats.

        Inputs: player_id (the raw player "id"), map_points (flattened).
        Output: list[_PlayerPoint], floors from 1; a point with no rooms
            or none of this player's stats is left out (its floor number
            still counts).
        Side effects: none.
        Exceptions: KeyError if a point lacks "rooms" or "player_stats", or
            a stats entry lacks "player_id".
        """
        result: list[_PlayerPoint] = []
        for floor, point in enumerate(map_points, start=1):
            own_stats = [
                stats
                for stats in point["player_stats"]
                if stats["player_id"] == player_id
            ]
            if not point["rooms"] or not own_stats:
                continue
            result.append(
                _PlayerPoint(floor, point["rooms"][0]["room_type"], own_stats[0])
            )
        return result

    def _build_deck_timeline(
        self, player: dict, points: list[_PlayerPoint]
    ) -> DeckTimeline:
        """Every copy the player ever held: the final deck (floor_left
        None), plus each removed copy and each transformed-away original
        (floor_left = the floor it left on).

        Inputs: player (raw, with "deck"), points.
        Output: DeckTimeline.
        Side effects: none.
        Exceptions: KeyError if a deck entry or a removed / transformed
            card lacks "id" or "floor_added_to_deck".
        """
        # The final deck: still held at run end
        cards = [
            TimedCard(entry["id"], entry["floor_added_to_deck"], None)
            for entry in player["deck"]
        ]

        # Copies that left: removed at a shop, or transformed into another
        for point in points:
            for removed in point.stats.get("cards_removed", []):
                cards.append(
                    TimedCard(
                        removed["id"], removed["floor_added_to_deck"], point.floor
                    )
                )
            for transform in point.stats.get("cards_transformed", []):
                original = transform["original_card"]
                cards.append(
                    TimedCard(
                        original["id"], original["floor_added_to_deck"], point.floor
                    )
                )
        return DeckTimeline(tuple(cards))

    def _choice_at(
        self,
        floor: int,
        options: list[dict],
        picked_index: int | None,
        timeline: DeckTimeline,
    ) -> CardRewardChoice:
        """Build the choice for one qualifying reward room.

        Inputs: floor, options (the raw card_choices), picked_index (the
            option taken, None = skipped), timeline.
        Output: CardRewardChoice with raw ids mapped through card_uuid_of.
        Side effects: as card_uuid_of.
        Exceptions: KeyError if an option lacks ["card"]["id"].
        """
        deck_before = tuple(
            self._card_uuid_of(raw_id) for raw_id in timeline.raw_card_ids_before(floor)
        )
        offered = tuple(self._card_uuid_of(option["card"]["id"]) for option in options)
        return CardRewardChoice(floor, deck_before, offered, picked_index)


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
