import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.draft_data_metrics.pick_sideboard_rate_metric import (
    PickSideboardRateMetric,
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

def _write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    header = "expansion,pack_number,pick_number,pick,pick_maindeck_rate,pick_sideboard_in_rate\n"
    lines = [header]
    for pick, sideboard_rate in rows:
        lines.append(f"MSH,0,0,{pick},0.9,{sideboard_rate}\n")
    path.write_text("".join(lines))

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

class TestAccumulateAndFinalize:
    def test_averages_sideboard_rate_per_card(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, rows=[("Bolt", 0.1), ("Bolt", 0.3)])
        output_path = tmp_path / "out.jsonl"
        metric = PickSideboardRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["metric_name"] == "pick_sideboard_rate"
        assert rows[0]["sample_size"] == 2
        assert rows[0]["value"] == (0.1 + 0.3) / 2

    def test_unresolved_pick_excluded_from_results(self, tmp_path: Path) -> None:
        binder = CardBinder()  # nothing registered
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, rows=[("Bolt", 0.1)])
        output_path = tmp_path / "out.jsonl"
        metric = PickSideboardRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        assert _read_jsonl(output_path) == []

    def test_multiple_accumulate_calls_add_not_overwrite(self, tmp_path: Path) -> None:
        # Regression test for the accumulation invariant this class's
        # own accumulate() docstring claims — matches v1's identical
        # coverage. A chunk_size of 1 forces CsvScanner to call
        # accumulate() once per row.
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "draft_data.csv"
        _write_csv(csv_path, rows=[("Bolt", 0.1), ("Bolt", 0.3)])
        output_path = tmp_path / "out.jsonl"
        metric = PickSideboardRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=1, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["sample_size"] == 2
        assert rows[0]["value"] == pytest.approx(0.2)

    def test_finalize_with_zero_resolved_cards_produces_a_valid_empty_file(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        output_path = tmp_path / "out.jsonl"
        metric = PickSideboardRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        result_path = metric.finalize()

        assert result_path == output_path
        assert _read_jsonl(output_path) == []

class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = (
            PickSideboardRateMetric.DEFAULT_OUTPUT_DIR
            / "MSH.PremierDraft.pick_sideboard_rate.jsonl"
        )
        assert PickSideboardRateMetric.default_output_path("MSH", "PremierDraft") == expected

    def test_varies_by_expansion_and_format(self) -> None:
        assert PickSideboardRateMetric.default_output_path(
            "MSH", "PremierDraft"
        ) != PickSideboardRateMetric.default_output_path("MSH", "TradDraft")

