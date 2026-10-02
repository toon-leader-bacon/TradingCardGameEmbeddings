"""Shared builders for the sts2_runs metric tests: a tiny StS2 binder and
raw runs in the spire_codex / sts2runs schema."""

import copy
import json
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)

KNOWN_CARD_IDS = ("STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE")


_BINDERS: dict[Path, CardBinder] = {}


def binder(tmp_path: Path) -> CardBinder:
    """A binder holding KNOWN_CARD_IDS (spire_codex aliases); the same
    instance per tmp_path, since ingestion mints random card uuids."""
    if tmp_path not in _BINDERS:
        _BINDERS[tmp_path] = _new_binder(tmp_path)
    return _BINDERS[tmp_path]


def _new_binder(tmp_path: Path) -> CardBinder:
    result = CardBinder()
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(
        json.dumps(
            [{"id": card_id, "name": card_id.title()} for card_id in KNOWN_CARD_IDS]
        ),
        encoding="utf-8",
    )
    SpireCodexCardIngestionStage().ingest(cards_path, result)
    return result


def _map_point(room_type: str, turns: int, damage: int, player_id: int = 1) -> dict:
    return {
        "map_point_type": room_type,
        "rooms": [{"room_type": room_type, "turns_taken": turns}],
        "player_stats": [
            {
                "player_id": player_id,
                "damage_taken": damage,
                "card_choices": (
                    [
                        {"card": {"id": "CARD.NEUTRALIZE"}, "was_picked": True},
                        {"card": {"id": "CARD.STRIKE_SILENT"}, "was_picked": False},
                    ]
                    if room_type == "monster"
                    else []
                ),
            }
        ],
    }


_BASE_RUN: dict = {
    "run_hash": "abc123",
    "win": False,
    "was_abandoned": False,
    "game_mode": "standard",
    "ascension": 3,
    "killed_by_encounter": "ENCOUNTER.THE_KIN_BOSS",
    "killed_by_event": "NONE.NONE",
    "players": [
        {
            "id": 1,
            "character": "CHARACTER.SILENT",
            "deck": [
                {"id": "CARD.STRIKE_SILENT", "floor_added_to_deck": 1},
                {
                    "id": "CARD.DEFEND_SILENT",
                    "floor_added_to_deck": 1,
                    "current_upgrade_level": 1,
                },
                {"id": "CARD.NEUTRALIZE", "floor_added_to_deck": 3},
                {"id": "CARD.NOT_IN_THE_BINDER", "floor_added_to_deck": 2},
            ],
            "relics": [{"id": "RELIC.A"}, {"id": "RELIC.B"}],
        }
    ],
    # Act 1: floors 1-2; act 2: floors 3-4 (the run dies in an elite)
    "map_point_history": [
        [_map_point("monster", 3, 5), _map_point("elite", 4, 10)],
        [_map_point("monster", 2, 7), _map_point("elite", 5, 30)],
    ],
}


def raw_run(**overrides: object) -> dict:
    """A lost, standard, two-act run (deep copy), with overrides applied."""
    result = copy.deepcopy(_BASE_RUN)
    result.update(overrides)
    return result
