"""Raw StS2 run dict -> Sts2Run (run_record.py): the one place the
sts2_runs metrics read the raw run schema.

RAW SHAPE (spire_codex's run export and sts2runs' dump share it; see
src/data_refinement/deck_box/sts2runs/extraction_stage.py): top-level
"win", "was_abandoned", "ascension", "game_mode", "killed_by_encounter"
/ "killed_by_event" ("NONE.NONE" when unset), "players" (each with "id",
"character", "deck", "relics") and "map_point_history" (one list of map
points per act; each map point has "rooms" and one "player_stats" entry
per player, matched by "player_id"). The run id key differs by source;
the parser takes it, and each deck's uuid, from that source's deck box
extraction stage, so a run's PlayerRun.deck_uuid is exactly the deck the
stage stored in the published deck box.

DERIVED STATS: the raw run has no stats block (unlike sts_gg), so the
run- and player-level numbers are summed from map_point_history:
    - total_turns: every room's "turns_taken";
    - total_combats: rooms whose room_type is monster, elite or boss;
    - elites_killed: elite rooms, less one if a lost run ended in one;
    - damage_taken / cards_picked / cards_skipped: the player's own
      player_stats entries ("damage_taken"; "card_choices" options with
      "was_picked" true / false).

CARD RESOLUTION: the same "CARD."-prefix strip + spire_codex alias
lookup the deck box stage uses, cached per raw id. A miss becomes a
CardSlot with card_uuid None (the box stored the Unknown sentinel there)
and is logged once per distinct id, not once per copy: over ~1.7M runs a
per-copy log would flood.
"""

import logging
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.sts2runs.extraction_stage import (
    Sts2RunsDeckExtractionStage,
)
from src.data_refinement.metrics.sts2_runs.run_record import (
    CardSlot,
    PlayerRun,
    RunOutcome,
    Sts2Run,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_CARD_ID_PREFIX = "CARD."
_NO_KILLER = "NONE.NONE"
_COMBAT_ROOM_TYPES = frozenset({"monster", "elite", "boss"})
_ELITE_ROOM_TYPE = "elite"


class Sts2RunParser:
    """Parses one source's raw runs into Sts2Runs (see module docstring)."""

    def __init__(
        self, card_lookup: CardLookup, stage: Sts2RunsDeckExtractionStage
    ) -> None:
        """
        Inputs:
            card_lookup: the Slay the Spire 2 binder (spire_codex cards);
                never written to.
            stage: the deck box extraction stage for this run source
                (Sts2RunsDeckExtractionStage or
                SpireCodexRunsDeckExtractionStage); supplies RUN_ID_KEY
                and deck_uuid().
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._stage = stage
        self._card_uuids: dict[str, UUID | None] = {}

    def parse(self, row: dict) -> Sts2Run:
        """Parse one raw run.

        Inputs: row (dict), one raw run JSON object.
        Output: Sts2Run.
        Side effects: caches each raw card id's lookup; logs the first
            miss of each card id.
        Exceptions: KeyError / TypeError / AttributeError / ValueError if
            row lacks, or mistypes, a field the module docstring's RAW
            SHAPE names.

        Example:
            >>> parser = Sts2RunParser(binder, SpireCodexRunsDeckExtractionStage())
            >>> parser.parse(row).outcome
            <RunOutcome.LOSS: 'loss'>
        """
        run_id = str(row[self._stage.RUN_ID_KEY])
        outcome = _outcome(row)
        map_points = [point for act in row["map_point_history"] for point in act]
        rooms = [room for point in map_points for room in point["rooms"]]
        return Sts2Run(
            run_id=run_id,
            game_mode=row["game_mode"],
            is_cheated=bool(row.get("_isCheated", False)),
            outcome=outcome,
            killed_by=_killed_by(row) if outcome is RunOutcome.LOSS else None,
            ascension=row["ascension"],
            floors_per_act=tuple(len(act) for act in row["map_point_history"]),
            total_turns=sum(room.get("turns_taken", 0) for room in rooms),
            total_combats=sum(
                room["room_type"] in _COMBAT_ROOM_TYPES for room in rooms
            ),
            elites_killed=_elites_killed(map_points, outcome),
            players=tuple(
                self._player_run(run_id, index, player, map_points)
                for index, player in enumerate(row["players"])
            ),
        )

    def _player_run(
        self, run_id: str, player_index: int, player: dict, map_points: list[dict]
    ) -> PlayerRun:
        """One raw player -> PlayerRun.

        Inputs: run_id, player_index (position in "players"), player (raw
            player dict), map_points (every act's map points, flattened).
        Output: PlayerRun. Side effects: as parse().
        Exceptions: KeyError if player lacks "id", "character", "deck" or
            "relics".
        """
        own_stats = [
            stats
            for point in map_points
            for stats in point["player_stats"]
            if stats["player_id"] == player["id"]
        ]
        choices = [
            choice for stats in own_stats for choice in stats.get("card_choices", [])
        ]
        picked = sum(bool(choice.get("was_picked")) for choice in choices)
        return PlayerRun(
            deck_uuid=self._stage.deck_uuid(run_id, player_index),
            character=player["character"],
            deck=tuple(self._card_slot(entry) for entry in player["deck"]),
            relic_count=len(player["relics"]),
            damage_taken=sum(stats.get("damage_taken", 0) for stats in own_stats),
            cards_picked=picked,
            cards_skipped=len(choices) - picked,
        )

    def _card_slot(self, entry: dict) -> CardSlot:
        """One raw deck entry -> CardSlot.

        Inputs: entry (dict with "id", "floor_added_to_deck" and an
            optional "current_upgrade_level").
        Output: CardSlot. Side effects: as parse().
        Exceptions: KeyError if entry lacks "id" or "floor_added_to_deck".
        """
        return CardSlot(
            card_uuid=self._card_uuid(entry["id"]),
            floor_added=entry["floor_added_to_deck"],
            upgraded=entry.get("current_upgrade_level", 0) > 0,
        )

    def _card_uuid(self, raw_card_id: str) -> UUID | None:
        """The nocab_uuid for a raw "CARD.<NAME>" id, or None when
        spire_codex has no such alias (see CARD RESOLUTION).

        Inputs: raw_card_id (str). Output: UUID | None.
        Side effects: caches the answer; logs the first miss of an id.
        Exceptions: none.
        """
        if raw_card_id not in self._card_uuids:
            card = self._card_lookup.get_by_alias(
                GameId.SLAY_THE_SPIRE_2,
                DataSource.SPIRE_CODEX,
                raw_card_id.removeprefix(_CARD_ID_PREFIX),
            )
            if card is None:
                _logger.warning(
                    "Sts2RunParser: no spire_codex card for %r; its slots are "
                    "the Unknown sentinel in the deck box and are skipped by "
                    "per-card metrics",
                    raw_card_id,
                )
            self._card_uuids[raw_card_id] = card.nocab_uuid if card else None
        return self._card_uuids[raw_card_id]


def _outcome(row: dict) -> RunOutcome:
    """Inputs: row (raw run). Output: RunOutcome. Side effects: none.
    Exceptions: KeyError if row lacks "win" or "was_abandoned"."""
    if row["was_abandoned"]:
        return RunOutcome.ABANDONED
    return RunOutcome.WIN if row["win"] else RunOutcome.LOSS


def _killed_by(row: dict) -> str | None:
    """The encounter that ended a lost run, else the event, else None.
    Inputs: row (raw run). Output: str | None. Side effects: none.
    Exceptions: none."""
    for key in ("killed_by_encounter", "killed_by_event"):
        killer = row.get(key, _NO_KILLER)
        if killer and killer != _NO_KILLER:
            return killer
    return None


def _elites_killed(map_points: list[dict], outcome: RunOutcome) -> int:
    """Elite rooms entered, less the one a lost run died in (a loss ends
    on its last map point).

    Inputs: map_points (flattened), outcome. Output: int.
    Side effects: none. Exceptions: none.
    """
    elites_entered = [
        any(room["room_type"] == _ELITE_ROOM_TYPE for room in point["rooms"])
        for point in map_points
    ]
    died_in_elite = (
        outcome is RunOutcome.LOSS and bool(elites_entered) and elites_entered[-1]
    )
    return sum(elites_entered) - int(died_in_elite)
