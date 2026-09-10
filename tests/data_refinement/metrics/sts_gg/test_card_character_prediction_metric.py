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
from src.data_refinement.metrics.sts_gg.card_character_prediction_metric import (
    CardCharacterPredictionMetric,
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


def _run_row(character: str, deck_card_ids: list[str]) -> dict:
    return {
        "character": character,
        "deck": [
            {"id": f"CARD.{card_id}", "upgraded": False, "floor": 1}
            for card_id in deck_card_ids
        ],
    }


class TestAccumulate:
    def test_single_run_gives_probability_one(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardCharacterPredictionMetric(
            binder, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_run_row("CHARACTER.SILENT", ["STRIKE_SILENT"]))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        row = df.iloc[0]
        assert row["nocab_uuid"] == str(_uuid_for(binder, "STRIKE_SILENT"))
        assert list(row["characters"]) == ["CHARACTER.SILENT"]
        assert list(row["probabilities"]) == pytest.approx([1.0])
        assert row["sample_count"] == 1

    def test_card_seen_with_two_characters_splits_probability(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardCharacterPredictionMetric(
            binder, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(_run_row("CHARACTER.SILENT", ["STRIKE_SILENT"]))
        metric.accumulate(_run_row("CHARACTER.SILENT", ["STRIKE_SILENT"]))
        metric.accumulate(_run_row("CHARACTER.REGENT", ["STRIKE_SILENT"]))
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 1
        row = df.iloc[0]
        by_character = dict(zip(row["characters"], row["probabilities"]))
        assert by_character["CHARACTER.SILENT"] == pytest.approx(2 / 3)
        assert by_character["CHARACTER.REGENT"] == pytest.approx(1 / 3)
        assert row["sample_count"] == 3

    def test_multiple_copies_in_one_run_are_independent_samples(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardCharacterPredictionMetric(
            binder, output_path=tmp_path / "out.parquet"
        )

        metric.accumulate(
            _run_row("CHARACTER.SILENT", ["STRIKE_SILENT", "STRIKE_SILENT"])
        )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert df.iloc[0]["sample_count"] == 2

    def test_unresolved_card_is_excluded_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        metric = CardCharacterPredictionMetric(
            binder, output_path=tmp_path / "out.parquet"
        )

        with caplog.at_level(logging.ERROR):
            metric.accumulate(
                _run_row("CHARACTER.SILENT", ["STRIKE_SILENT", "NOT_A_REAL_CARD"])
            )
        metric.finalize()

        df = pd.read_parquet(tmp_path / "out.parquet")
        assert len(df) == 1
        assert any(
            record.levelno == logging.ERROR and "NOT_A_REAL_CARD" in record.getMessage()
            for record in caplog.records
        )


class TestFinalize:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        binder = _binder_from_rows(
            [{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path
        )
        nested_path = tmp_path / "nested" / "dir" / "out.parquet"
        metric = CardCharacterPredictionMetric(binder, output_path=nested_path)
        metric.accumulate(_run_row("CHARACTER.SILENT", ["STRIKE_SILENT"]))

        output_path = metric.finalize()

        assert output_path == nested_path
        assert nested_path.exists()

    def test_no_cards_seen_writes_empty_file(self, tmp_path: Path) -> None:
        binder = _binder_from_rows([], tmp_path)
        metric = CardCharacterPredictionMetric(
            binder, output_path=tmp_path / "out.parquet"
        )

        output_path = metric.finalize()

        df = pd.read_parquet(output_path)
        assert len(df) == 0


def test_accepts_and_ignores_a_deck_box(tmp_path: Path) -> None:
    from src.data_refinement.deck_box.deck_box import DeckBox

    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    deck_box = DeckBox()
    metric = CardCharacterPredictionMetric(
        binder, deck_box=deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_run_row("CHARACTER.SILENT", ["STRIKE_SILENT"]))
    metric.finalize()

    assert list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2)) == []


def test_default_output_path() -> None:
    assert CardCharacterPredictionMetric.DEFAULT_OUTPUT_PATH == Path(
        "data/metrics/sts_gg/card_character_prediction.parquet"
    )
