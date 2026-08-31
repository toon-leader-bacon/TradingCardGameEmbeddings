import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.replay_data_metrics.tutor_target_rate_metric import (
    TutorTargetRateMetric,
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
    columns = ["user_turn_1_cards_tutored", "user_turn_2_cards_tutored"]
    lines = [",".join(columns) + "\n"]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns) + "\n")
    path.write_text("".join(lines))


class TestAccumulateAndFinalize:
    def test_computes_share_of_all_tutor_events(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = _register_arena_card(binder, "Bolt", "1")
        bear = _register_arena_card(binder, "Bear", "2")
        csv_path = tmp_path / "replay.csv"
        _write_csv(
            csv_path,
            rows=[
                {"user_turn_1_cards_tutored": "1|2"},
                {"user_turn_2_cards_tutored": "1"},
            ],
        )
        output_path = tmp_path / "out.jsonl"
        metric = TutorTargetRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = {row["nocab_uuid"]: row for row in _read_jsonl(output_path)}
        assert len(rows) == 2
        assert rows[str(bolt.nocab_uuid)]["sample_size"] == 2
        assert rows[str(bolt.nocab_uuid)]["value"] == 2 / 3
        assert rows[str(bear.nocab_uuid)]["sample_size"] == 1
        assert rows[str(bear.nocab_uuid)]["value"] == 1 / 3

    def test_default_output_path(self) -> None:
        expected = (
            TutorTargetRateMetric.DEFAULT_OUTPUT_DIR / "MSH.PremierDraft.tutor_target_rate.jsonl"
        )
        assert TutorTargetRateMetric.default_output_path("MSH", "PremierDraft") == expected

    def test_finalize_with_no_accumulate_calls_produces_valid_empty_file(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        output_path = tmp_path / "out.jsonl"
        metric = TutorTargetRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        assert metric.finalize() == output_path
        assert _read_jsonl(output_path) == []
