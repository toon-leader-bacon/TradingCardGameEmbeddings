import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.replay_data_metrics.average_turns_remaining_post_cast_metric import (  # noqa: E501 -- module path is one unbreakable dotted identifier
    AverageTurnsRemainingPostCastMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.seventeenlands.v2.replay_data_metrics.cast_column_fixture import (
    write_cast_csv,
)

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

class TestAccumulateAndFinalize:
    def test_computes_average_turns_remaining(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = _register_arena_card(binder, "Bolt", "1")
        csv_path = tmp_path / "replay.csv"
        write_cast_csv(
            csv_path,
            base_columns=["num_turns"],
            rows=[
                {"num_turns": 5, "user_turn_2_creatures_cast": "1"},
                {"num_turns": 6, "user_turn_4_creatures_cast": "1"},
            ],
        )
        output_path = tmp_path / "out.jsonl"
        metric = AverageTurnsRemainingPostCastMetric(
            binder, GameId.MTG, "MSH", "PremierDraft", output_path
        )

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["sample_size"] == 2
        assert rows[0]["value"] == 2.5

    def test_default_output_path(self) -> None:
        expected = (
            AverageTurnsRemainingPostCastMetric.DEFAULT_OUTPUT_DIR
            / "MSH.PremierDraft.average_turns_remaining_post_cast.jsonl"
        )
        assert (
            AverageTurnsRemainingPostCastMetric.default_output_path("MSH", "PremierDraft")
            == expected
        )

    def test_finalize_with_no_accumulate_calls_produces_valid_empty_file(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        output_path = tmp_path / "out.jsonl"
        metric = AverageTurnsRemainingPostCastMetric(
            binder, GameId.MTG, "MSH", "PremierDraft", output_path
        )

        assert metric.finalize() == output_path
        assert _read_jsonl(output_path) == []

