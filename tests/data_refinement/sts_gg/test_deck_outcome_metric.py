import json
import logging
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)
from src.data_refinement.sts_gg.deck_outcome_metric import (
    DeckOutcomeMetric,
    scan_runs_jsonl,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _binder_from_rows(rows: list[dict], tmp_path: Path) -> CardBinder:
    binder = CardBinder()
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(json.dumps(rows), encoding="utf-8")
    SpireCodexCardIngestionStage().ingest(cards_path, binder)
    return binder


def _uuid_for(binder: CardBinder, card_id: str) -> UUID:
    card = binder.get_by_alias(GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, card_id)
    assert card is not None
    return card.nocab_uuid


def _run_row(run_id: str, card_ids: list[str], win: bool) -> dict:
    return {
        "id": run_id,
        "win": win,
        "deck": [{"id": f"CARD.{card_id}", "upgraded": False, "floor": 1} for card_id in card_ids],
    }


class TestAccumulate:
    def test_every_card_resolves(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}, {"id": "DEFEND_SILENT", "name": "Defend"}],
            tmp_path,
        )
        metric = DeckOutcomeMetric(binder, tmp_path / "out.parquet")

        metric.accumulate(_run_row("run1", ["STRIKE_SILENT", "DEFEND_SILENT"], win=True))
        metric.finalize()

        rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
        assert rows == [
            {
                "run_id": "run1",
                "deck_nocab_uuids": [
                    str(_uuid_for(binder, "STRIKE_SILENT")),
                    str(_uuid_for(binder, "DEFEND_SILENT")),
                ],
                "win": True,
            }
        ]

    def test_unresolved_card_is_excluded_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
        metric = DeckOutcomeMetric(binder, tmp_path / "out.parquet")

        with caplog.at_level(logging.ERROR):
            metric.accumulate(_run_row("run1", ["STRIKE_SILENT", "NOT_A_REAL_CARD"], win=False))
        metric.finalize()

        rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
        assert rows == [
            {
                "run_id": "run1",
                "deck_nocab_uuids": [str(_uuid_for(binder, "STRIKE_SILENT"))],
                "win": False,
            }
        ]
        assert any(
            record.levelno == logging.ERROR and "NOT_A_REAL_CARD" in record.getMessage()
            for record in caplog.records
        )

    def test_multiple_copies_of_a_card_are_preserved(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
        metric = DeckOutcomeMetric(binder, tmp_path / "out.parquet")

        metric.accumulate(_run_row("run1", ["STRIKE_SILENT", "STRIKE_SILENT"], win=True))
        metric.finalize()

        rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
        expected_uuid = str(_uuid_for(binder, "STRIKE_SILENT"))
        assert rows[0]["deck_nocab_uuids"] == [expected_uuid, expected_uuid]


class TestFinalize:
    def test_is_idempotent(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
        metric = DeckOutcomeMetric(binder, tmp_path / "out.parquet")
        metric.accumulate(_run_row("run1", ["STRIKE_SILENT"], win=True))

        first_path = metric.finalize()
        second_path = metric.finalize()

        assert first_path == second_path == tmp_path / "out.parquet"
        assert pq.read_table(first_path).num_rows == 1


class TestScanRunsJsonl:
    def test_drives_every_line_into_one_parquet_output(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}, {"id": "DEFEND_SILENT", "name": "Defend"}],
            tmp_path,
        )
        raw_path = tmp_path / "runs.jsonl"
        raw_path.write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    _run_row("run1", ["STRIKE_SILENT"], win=True),
                    _run_row("run2", ["DEFEND_SILENT"], win=False),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        metric = DeckOutcomeMetric(binder, tmp_path / "out.parquet")

        output_path = scan_runs_jsonl(raw_path, metric)

        rows = pq.read_table(output_path).to_pylist()
        assert [row["run_id"] for row in rows] == ["run1", "run2"]
        assert [row["win"] for row in rows] == [True, False]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
        raw_path = tmp_path / "runs.jsonl"
        raw_path.write_text(
            json.dumps(_run_row("run1", ["STRIKE_SILENT"], win=True)) + "\n\n   \n",
            encoding="utf-8",
        )
        metric = DeckOutcomeMetric(binder, tmp_path / "out.parquet")

        output_path = scan_runs_jsonl(raw_path, metric)

        assert pq.read_table(output_path).num_rows == 1

    def test_still_finalizes_when_a_line_is_malformed(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
        raw_path = tmp_path / "runs.jsonl"
        raw_path.write_text(
            json.dumps(_run_row("run1", ["STRIKE_SILENT"], win=True))
            + "\n"
            + "{not valid json"
            + "\n",
            encoding="utf-8",
        )
        metric = DeckOutcomeMetric(binder, tmp_path / "out.parquet")

        with pytest.raises(json.JSONDecodeError):
            scan_runs_jsonl(raw_path, metric)

        # The writer was still closed (via scan_runs_jsonl's try/finally),
        # so the file has a valid footer and is readable, even though the
        # malformed line's run (and anything after it) never got written.
        rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
        assert [row["run_id"] for row in rows] == ["run1"]
