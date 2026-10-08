"""Typed records for one Slay the Spire 2 run, as the sts2_runs metrics
read it.

Both run sources (spire_codex's run export and sts2runs' dump) share one
raw schema; run_parser.py turns a raw run dict into these records, and
nothing past that parse step touches the raw dict. Every value a metric
needs is computed once, at parse time, from the run's top-level fields,
its players' final decks and relics, and its map_point_history.

Card identity: a CardSlot's card_uuid is None when the card id has no
spire_codex alias. The published deck box stores the Unknown sentinel
card in that slot instead; the metrics never emit a per-card row for it.
"""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

# Room types that are combats; the rooms whose rewards include a card choice.
COMBAT_ROOM_TYPES = frozenset({"monster", "elite", "boss"})


class RunOutcome(Enum):
    """How a run ended. A raw run is abandoned or not, and won or not;
    no run in the data is both won and abandoned, so three cases."""

    WIN = "win"
    LOSS = "loss"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class CardSlot:
    """One copy of a card in a player's final deck.

    card_uuid: the card's nocab_uuid, or None if it has no spire_codex
        alias (the deck box holds the Unknown sentinel there).
    floor_added: the floor the copy entered the deck (1 = run start).
    upgraded: whether the copy ended the run upgraded.
    """

    card_uuid: UUID | None
    floor_added: int
    upgraded: bool


@dataclass(frozen=True)
class CardRewardChoice:
    """One card-reward room as the player faced it: the deck on arrival,
    the cards offered, and which one (if any) was taken.

    floor: the floor of the reward (deck_timeline.py's numbering).
    deck_before: the deck on arrival, one card_uuid per copy (None = no
        spire_codex alias). Card identity only: no upgrade or enchantment.
    offered: the cards offered, in the order shown (None = no alias).
    picked_index: the offered card taken, or None if the player skipped.
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


@dataclass(frozen=True)
class PlayerRun:
    """One player's side of a run (co-op runs have several).

    deck_uuid: the nocab_uuid the deck box extraction stage gave this
        (run, player) deck in the published Slay the Spire 2 deck box.
    character: the raw "CHARACTER.<NAME>" id.
    deck: the final deck, one slot per copy.
    relic_count: relics held at run end.
    damage_taken: damage taken over every map point.
    cards_picked / cards_skipped: card-reward options taken / not taken,
        summed over every card choice the player saw.
    card_rewards: each qualifying card-reward room, in floor order (see
        card_reward_reader.py).
    """

    deck_uuid: UUID
    character: str
    deck: tuple[CardSlot, ...]
    relic_count: int
    damage_taken: int
    cards_picked: int
    cards_skipped: int
    card_rewards: tuple[CardRewardChoice, ...]

    def known_card_uuids(self) -> list[UUID]:
        """The deck's card uuids, one per copy, without unaliased cards.

        Inputs: none. Output: list[UUID]. Side effects: none.
        Exceptions: none.

        Example:
            >>> len(player.known_card_uuids()) <= len(player.deck)
            True
        """
        return [slot.card_uuid for slot in self.deck if slot.card_uuid is not None]


@dataclass(frozen=True)
class Sts2Run:
    """One run, parsed.

    run_id: the source's own run id (str(_serverId) or run_hash).
    game_mode: raw "game_mode" ("standard", "daily", "custom").
    is_cheated: sts2runs' _isCheated flag (False when absent).
    outcome: see RunOutcome.
    killed_by: the encounter (or event) id that ended a LOSS; None for
        any other outcome, and for a loss the raw run names no killer.
    ascension: the ascension level played.
    floors_per_act: map points visited in each act, in order.
    total_turns: combat turns summed over every room.
    total_combats: monster/elite/boss rooms entered.
    elites_killed: elite rooms entered and survived.
    players: one PlayerRun per raw player, in raw order.
    """

    run_id: str
    game_mode: str
    is_cheated: bool
    outcome: RunOutcome
    killed_by: str | None
    ascension: int
    floors_per_act: tuple[int, ...]
    total_turns: int
    total_combats: int
    elites_killed: int
    players: tuple[PlayerRun, ...]

    @property
    def win(self) -> bool:
        """Whether the run was won.
        Output: bool. Side effects: none. Exceptions: none."""
        return self.outcome is RunOutcome.WIN

    @property
    def floors_cleared(self) -> int:
        """Floors visited, less the floor a lost run ended on.
        Output: int. Side effects: none. Exceptions: none."""
        floors = sum(self.floors_per_act)
        return floors - 1 if self.outcome is RunOutcome.LOSS and floors else floors

    @property
    def act_2_start_floor(self) -> int | None:
        """The first floor of act 2, or None if the run never reached it.

        Floors count up across acts, so act 2 starts one floor after
        act 1's last map point; the act lengths vary run to run, so this
        is per run, never a fixed floor.

        Output: int | None. Side effects: none. Exceptions: none.

        Example:
            >>> run.floors_per_act, run.act_2_start_floor
            ((17, 8), 18)
        """
        if len(self.floors_per_act) < 2:
            return None
        return self.floors_per_act[0] + 1
