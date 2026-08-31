import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.replay_data_metrics.combat_kill_involvement_rate_metric import (  # noqa: E501 -- module path is one unbreakable dotted identifier
    CombatKillInvolvementRateMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _register_arena_card(binder: CardBinder, name: str, arena_id: str) -> GenericCard:
    stored = binder.add(_card(name)).stored_card
    binder.register_alias(GameId.MTG, DataSource.ARENA, arena_id, stored.nocab_uuid)
    return stored


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "user_turn_1_creatures_attacked",
        "user_turn_1_user_creatures_killed_combat",
        "oppo_turn_1_user_creatures_killed_combat",
        "oppo_turn_2_creatures_blocked",
        "oppo_turn_2_oppo_creatures_killed_combat",
    ]
    lines = [",".join(columns) + "\n"]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns) + "\n")
    path.write_text("".join(lines))


class TestAccumulateAndFinalize:
    def test_computes_kill_involvement_rate(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = _register_arena_card(binder, "Bolt", "1")
        bear = _register_arena_card(binder, "Bear", "2")
        csv_path = tmp_path / "replay.csv"
        _write_csv(
            csv_path,
            rows=[
                {
                    "user_turn_1_creatures_attacked": "1",
                    "user_turn_1_user_creatures_killed_combat": "1",
                    "oppo_turn_2_creatures_blocked": "2",
                }
            ],
        )
        output_path = tmp_path / "out.jsonl"
        metric = CombatKillInvolvementRateMetric(
            binder, GameId.MTG, "MSH", "PremierDraft", output_path
        )

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = {row["nocab_uuid"]: row for row in _read_jsonl(output_path)}
        assert len(rows) == 2
        assert rows[str(bolt.nocab_uuid)]["sample_size"] == 1
        assert rows[str(bolt.nocab_uuid)]["value"] == 1.0
        assert rows[str(bear.nocab_uuid)]["sample_size"] == 1
        assert rows[str(bear.nocab_uuid)]["value"] == 0.0

    def test_kill_on_the_other_sides_same_numbered_turn_does_not_count(
        self, tmp_path: Path
    ) -> None:
        """<side>_turn_<N> is a per-side turn counter (see this
        metric's own module docstring, and cast_event_scanner.py's
        CastEvent.turn docstring) — user_turn_1 and oppo_turn_1 are not
        reliably the same real turn, so a same-side (own-turn/own-side)
        kill recorded under the OTHER side's same-numbered turn must
        not count as involvement.
        """
        binder = CardBinder()
        bolt = _register_arena_card(binder, "Bolt", "1")
        csv_path = tmp_path / "replay.csv"
        _write_csv(
            csv_path,
            rows=[
                {
                    "user_turn_1_creatures_attacked": "1",
                    "oppo_turn_2_oppo_creatures_killed_combat": "1",
                }
            ],
        )
        output_path = tmp_path / "out.jsonl"
        metric = CombatKillInvolvementRateMetric(
            binder, GameId.MTG, "MSH", "PremierDraft", output_path
        )

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = {row["nocab_uuid"]: row for row in _read_jsonl(output_path)}
        assert rows[str(bolt.nocab_uuid)]["sample_size"] == 1
        assert rows[str(bolt.nocab_uuid)]["value"] == 0.0

    def test_default_output_path(self) -> None:
        expected = (
            CombatKillInvolvementRateMetric.DEFAULT_OUTPUT_DIR
            / "MSH.PremierDraft.combat_kill_involvement_rate.jsonl"
        )
        assert (
            CombatKillInvolvementRateMetric.default_output_path("MSH", "PremierDraft") == expected
        )

    def test_finalize_with_no_accumulate_calls_produces_valid_empty_file(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        output_path = tmp_path / "out.jsonl"
        metric = CombatKillInvolvementRateMetric(
            binder, GameId.MTG, "MSH", "PremierDraft", output_path
        )

        assert metric.finalize() == output_path
        assert _read_jsonl(output_path) == []
