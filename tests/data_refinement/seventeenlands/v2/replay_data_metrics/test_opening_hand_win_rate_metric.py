import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.replay_data_metrics.opening_hand_win_rate_metric import (
    OpeningHandWinRateMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = "won,num_mulligans,opening_hand,candidate_hand_1\n"

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

def _write_csv(path: Path, rows: list[tuple[bool, str]]) -> None:
    lines = [_HEADER]
    for won, opening_hand in rows:
        lines.append(f"{won},0,{opening_hand},\n")
    path.write_text("".join(lines))

class TestAccumulateAndFinalize:
    def test_computes_wins_and_games(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = _register_arena_card(binder, "Bolt", "1")
        csv_path = tmp_path / "replay.csv"
        _write_csv(csv_path, rows=[(True, "1"), (True, "1"), (False, "1")])
        output_path = tmp_path / "out.jsonl"
        metric = OpeningHandWinRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["sample_size"] == 3
        assert rows[0]["value"] == 2 / 3

    def test_default_output_path(self) -> None:
        expected = (
            OpeningHandWinRateMetric.DEFAULT_OUTPUT_DIR
            / "MSH.PremierDraft.opening_hand_win_rate.jsonl"
        )
        assert OpeningHandWinRateMetric.default_output_path("MSH", "PremierDraft") == expected

    def test_finalize_with_no_accumulate_calls_produces_valid_empty_file(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        output_path = tmp_path / "out.jsonl"
        metric = OpeningHandWinRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        assert metric.finalize() == output_path
        assert _read_jsonl(output_path) == []

