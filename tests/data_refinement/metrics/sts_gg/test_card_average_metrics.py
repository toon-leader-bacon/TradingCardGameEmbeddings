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
from src.data_refinement.metrics.sts_gg.card_average_metric import CardAverageMetric
from src.data_refinement.metrics.sts_gg.card_average_metrics import (
    CardDeckSizeMetric,
    CardElitesKilledMetric,
    CardFloorsClearedMetric,
    CardRelicCountMetric,
    CardTotalCardsPickedMetric,
    CardTotalCombatsMetric,
    CardTotalDamageTakenMetric,
    CardTotalTurnsMetric,
    CardWinRateMetric,
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


def _run_row(deck_card_ids: list[str], **overrides: object) -> dict:
    row: dict = {
        "win": True,
        "relicCount": 8,
        "deckSize": 6,
        "stats": {
            "totalDamageTaken": 55,
            "totalCardsPicked": 3,
            "totalTurns": 23,
            "elitesKilled": 0,
            "floorsCleared": 48,
            "totalCombats": 20,
        },
        "deck": [
            {"id": f"CARD.{card_id}", "upgraded": False, "floor": 1}
            for card_id in deck_card_ids
        ],
    }
    row.update(overrides)
    return row


# (metric class, expected value for _run_row()'s defaults)
_CASES: list[tuple[type[CardAverageMetric], float]] = [
    (CardRelicCountMetric, 8.0),
    (CardTotalDamageTakenMetric, 55.0),
    (CardDeckSizeMetric, 6.0),
    (CardTotalCardsPickedMetric, 3.0),
    (CardTotalTurnsMetric, 23.0),
    (CardElitesKilledMetric, 0.0),
    (CardFloorsClearedMetric, 48.0),
    (CardTotalCombatsMetric, 20.0),
    (CardWinRateMetric, 1.0),
]


@pytest.mark.parametrize("metric_cls,expected_value", _CASES)
def test_single_run_average_equals_that_runs_value(
    metric_cls: type[CardAverageMetric], expected_value: float, tmp_path: Path
) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    metric = metric_cls(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(_run_row(["STRIKE_SILENT"]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    row = df.iloc[0]
    assert row["nocab_uuid"] == str(_uuid_for(binder, "STRIKE_SILENT"))
    assert row[metric_cls.LABEL_COLUMN] == expected_value
    assert row["sample_count"] == 1


@pytest.mark.parametrize("metric_cls,_expected_value", _CASES)
def test_multiple_copies_in_one_run_are_independent_samples(
    metric_cls: type[CardAverageMetric], _expected_value: float, tmp_path: Path
) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    metric = metric_cls(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(_run_row(["STRIKE_SILENT", "STRIKE_SILENT"]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.iloc[0]["sample_count"] == 2


def test_win_rate_averages_across_wins_and_losses(tmp_path: Path) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    metric = CardWinRateMetric(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(_run_row(["STRIKE_SILENT"], win=True))
    metric.accumulate(_run_row(["STRIKE_SILENT"], win=False))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    row = df.iloc[0]
    assert row["win_rate"] == 0.5
    assert row["sample_count"] == 2


def test_relic_count_averages_across_runs(tmp_path: Path) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    metric = CardRelicCountMetric(binder, output_path=tmp_path / "out.parquet")

    metric.accumulate(_run_row(["STRIKE_SILENT"], relicCount=4))
    metric.accumulate(_run_row(["STRIKE_SILENT"], relicCount=10))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    row = df.iloc[0]
    assert row["average_relic_count"] == 7.0
    assert row["sample_count"] == 2


def test_unresolved_card_is_excluded_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    metric = CardRelicCountMetric(binder, output_path=tmp_path / "out.parquet")

    with caplog.at_level(logging.ERROR):
        metric.accumulate(_run_row(["STRIKE_SILENT", "NOT_A_REAL_CARD"]))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 1
    assert any(
        record.levelno == logging.ERROR and "NOT_A_REAL_CARD" in record.getMessage()
        for record in caplog.records
    )


def test_accepts_and_ignores_a_deck_box(tmp_path: Path) -> None:
    from src.data_refinement.deck_box.deck_box import DeckBox

    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    deck_box = DeckBox()
    metric = CardRelicCountMetric(
        binder, deck_box=deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_run_row(["STRIKE_SILENT"]))
    metric.finalize()

    assert list(deck_box.all_decks(GameId.SLAY_THE_SPIRE_2)) == []


@pytest.mark.parametrize("metric_cls,_expected_value", _CASES)
def test_no_cards_seen_writes_empty_file(
    metric_cls: type[CardAverageMetric], _expected_value: float, tmp_path: Path
) -> None:
    binder = _binder_from_rows([], tmp_path)
    metric = metric_cls(binder, output_path=tmp_path / "out.parquet")

    output_path = metric.finalize()

    df = pd.read_parquet(output_path)
    assert len(df) == 0


@pytest.mark.parametrize("metric_cls,_expected_value", _CASES)
def test_default_output_path_is_under_metrics_sts_gg(
    metric_cls: type[CardAverageMetric], _expected_value: float
) -> None:
    assert metric_cls.DEFAULT_OUTPUT_PATH.parent == Path("data/metrics/sts_gg")
    assert metric_cls.DEFAULT_OUTPUT_PATH.suffix == ".parquet"


def test_default_output_paths_are_all_distinct() -> None:
    output_paths = {metric_cls.DEFAULT_OUTPUT_PATH for metric_cls, _ in _CASES}
    assert len(output_paths) == len(_CASES)
