import json
import logging
from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.sts_gg.card_upgrade_rate_metric import (
    CardUpgradeRateMetric,
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


def _deck_row(entries: list[tuple[str, bool]]) -> dict:
    return {
        "deck": [
            {"id": f"CARD.{card_id}", "upgraded": upgraded, "floor": 1}
            for card_id, upgraded in entries
        ]
    }


def test_accepts_and_ignores_a_deck_box(tmp_path: Path) -> None:
    # deck_box exists only for constructor-shape consistency with this
    # container's multi-card metrics - this metric is per-card and
    # must not read from or write into it.
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    deck_box = DeckBox()
    metric = CardUpgradeRateMetric(
        binder, deck_box=deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_deck_row([("STRIKE_SILENT", True)]))
    metric.finalize()

    assert list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2)) == []


class TestAccumulate:
    def test_tallies_upgraded_and_total_per_card(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardUpgradeRateMetric(binder, output_path=tmp_path / "out.parquet")

        metric.accumulate(_deck_row([("STRIKE_SILENT", True)]))
        metric.accumulate(_deck_row([("STRIKE_SILENT", False)]))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        row = df.iloc[0]
        assert row["nocab_uuid"] == str(_uuid_for(binder, "STRIKE_SILENT"))
        assert row["upgrade_rate"] == 0.5
        assert row["sample_count"] == 2

    def test_multiple_copies_in_one_run_are_independent_samples(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardUpgradeRateMetric(binder, output_path=tmp_path / "out.parquet")

        metric.accumulate(
            _deck_row([("STRIKE_SILENT", True), ("STRIKE_SILENT", False)])
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        row = df.iloc[0]
        assert row["sample_count"] == 2
        assert row["upgrade_rate"] == 0.5

    def test_unresolved_card_is_excluded_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardUpgradeRateMetric(binder, output_path=tmp_path / "out.parquet")

        with caplog.at_level(logging.ERROR):
            metric.accumulate(
                _deck_row([("STRIKE_SILENT", True), ("NOT_A_REAL_CARD", True)])
            )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 1
        assert any(
            record.levelno == logging.ERROR and "NOT_A_REAL_CARD" in record.getMessage()
            for record in caplog.records
        )


class TestFinalize:
    def test_writes_one_row_per_card(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [
                {"id": "STRIKE_SILENT", "name": "Strike"},
                {"id": "DEFEND_SILENT", "name": "Defend"},
            ],
            tmp_path,
        )
        metric = CardUpgradeRateMetric(binder, output_path=tmp_path / "out.parquet")
        metric.accumulate(
            _deck_row([("STRIKE_SILENT", True), ("DEFEND_SILENT", False)])
        )

        output_path = metric.finalize()

        df = pd.read_parquet(output_path)
        assert set(df["nocab_uuid"]) == {
            str(_uuid_for(binder, "STRIKE_SILENT")),
            str(_uuid_for(binder, "DEFEND_SILENT")),
        }

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = CardUpgradeRateMetric(binder, output_path=nested_path)
        metric.accumulate(_deck_row([("STRIKE_SILENT", True)]))

        output_path = metric.finalize()

        assert output_path == nested_path
        assert nested_path.exists()

    def test_no_cards_seen_writes_empty_file(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([], tmp_path)
        metric = CardUpgradeRateMetric(binder, output_path=tmp_path / "out.parquet")

        output_path = metric.finalize()

        df = pd.read_parquet(output_path)
        assert len(df) == 0


def test_default_output_path() -> None:
    assert CardUpgradeRateMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/sts_gg/card_upgrade_rate.parquet"
    )
