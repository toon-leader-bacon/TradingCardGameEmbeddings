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
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.sts_gg.ascension_prediction_metric import (
    AscensionPredictionMetric,
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


def _run_row(run_id: str, ascension: int, deck_card_ids: list[str]) -> dict:
    return {
        "id": run_id,
        "ascension": ascension,
        "deck": [
            {"id": f"CARD.{card_id}", "upgraded": False, "floor": 1}
            for card_id in deck_card_ids
        ],
    }


class TestAccumulate:
    def test_writes_one_row_per_run(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

        metric.accumulate(_run_row("run1", 5, ["STRIKE_SILENT"]))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        assert table.num_rows == 1
        row = table.to_pylist()[0]
        assert row["run_id"] == "run1"
        assert row["ascension"] == 5
        expected_deck_uuid = deck_uuid_from_cards([_uuid_for(binder, "STRIKE_SILENT")])
        assert row["deck_uuid"] == str(expected_deck_uuid)

    def test_multiple_runs_write_multiple_rows(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

        metric.accumulate(_run_row("run1", 0, ["STRIKE_SILENT"]))
        metric.accumulate(_run_row("run2", 20, ["STRIKE_SILENT"]))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        assert table.num_rows == 2
        assert set(table.column("run_id").to_pylist()) == {"run1", "run2"}

    def test_identical_final_decks_share_one_deck_box_entry(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

        metric.accumulate(_run_row("run1", 0, ["STRIKE_SILENT"]))
        metric.accumulate(_run_row("run2", 20, ["STRIKE_SILENT"]))
        metric.finalize()

        table = pq.read_table(tmp_path / "out.parquet")
        deck_uuids = set(table.column("deck_uuid").to_pylist())
        assert len(deck_uuids) == 1
        assert len(list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2))) == 1

    def test_deck_box_entry_holds_the_resolved_card_multiset(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

        metric.accumulate(_run_row("run1", 3, ["STRIKE_SILENT", "STRIKE_SILENT"]))
        metric.finalize()

        [deck] = list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2))
        assert len(deck.card_nocab_uuids) == 2

    def test_empty_deck_writes_empty_deck_box_entry(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([], tmp_path)
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

        metric.accumulate(_run_row("run1", 0, []))
        metric.finalize()

        [deck] = list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2))
        assert deck.card_nocab_uuids == []

    def test_unresolved_card_is_excluded_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

        with caplog.at_level(logging.ERROR):
            metric.accumulate(_run_row("run1", 1, ["STRIKE_SILENT", "NOT_A_REAL_CARD"]))
        metric.finalize()

        [deck] = list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2))
        assert deck.card_nocab_uuids == [_uuid_for(binder, "STRIKE_SILENT")]
        assert any(
            record.levelno == logging.ERROR and "NOT_A_REAL_CARD" in record.getMessage()
            for record in caplog.records
        )


class TestFinalize:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        deck_box = DeckBox()
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = AscensionPredictionMetric(binder, deck_box, nested_path)
        metric.accumulate(_run_row("run1", 0, ["STRIKE_SILENT"]))

        output_path = metric.finalize()

        assert output_path == nested_path
        assert nested_path.exists()

    def test_is_idempotent(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")
        metric.accumulate(_run_row("run1", 0, ["STRIKE_SILENT"]))

        first_path = metric.finalize()
        second_path = metric.finalize()

        assert first_path == second_path

    def test_no_runs_seen_writes_empty_file(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([], tmp_path)
        deck_box = DeckBox()
        metric = AscensionPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

        output_path = metric.finalize()

        table = pq.read_table(output_path)
        assert table.num_rows == 0


def test_default_output_path() -> None:
    assert AscensionPredictionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/sts_gg/ascension_prediction.parquet"
    )
