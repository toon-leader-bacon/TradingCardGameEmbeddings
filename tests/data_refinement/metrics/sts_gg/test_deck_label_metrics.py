import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.sts_gg.deck_label_metric import DeckLabelMetric
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
    ElitesKilledMetric,
    FloorsClearedMetric,
    KilledByMetric,
    RelicCountMetric,
    TotalCardsPickedMetric,
    TotalCardsSkippedMetric,
    TotalCombatsMetric,
    TotalDamageTakenMetric,
    TotalTurnsMetric,
    WinMetric,
)


def _binder_from_rows(rows: list[dict], tmp_path: Path) -> CardBinder:
    binder = CardBinder()
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(json.dumps(rows), encoding="utf-8")
    SpireCodexCardIngestionStage().ingest(cards_path, binder)
    return binder


def _run_row(**overrides: object) -> dict:
    row: dict = {
        "id": "run1",
        "win": True,
        "character": "CHARACTER.SILENT",
        "relicCount": 8,
        "killedBy": None,
        "deck": [{"id": "CARD.STRIKE_SILENT", "upgraded": False, "floor": 1}],
        "stats": {
            "totalDamageTaken": 55,
            "totalCardsPicked": 3,
            "totalCardsSkipped": 116,
            "floorsCleared": 48,
            "totalCombats": 20,
            "totalTurns": 23,
            "elitesKilled": 0,
        },
    }
    row.update(overrides)
    return row


# (metric class, expected label value for _run_row()'s defaults)
_CASES: list[tuple[type[DeckLabelMetric], object]] = [
    (RelicCountMetric, 8),
    (CharacterPredictionMetric, "CHARACTER.SILENT"),
    (TotalDamageTakenMetric, 55),
    (TotalCardsPickedMetric, 3),
    (TotalCardsSkippedMetric, 116),
    (TotalTurnsMetric, 23),
    (ElitesKilledMetric, 0),
    (FloorsClearedMetric, 48),
    (TotalCombatsMetric, 20),
    (KilledByMetric, None),
    (WinMetric, True),
]


@pytest.mark.parametrize("metric_cls,expected_label", _CASES)
def test_writes_expected_label_for_its_column(
    metric_cls: type[DeckLabelMetric], expected_label: object, tmp_path: Path
) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    deck_box = DeckBox()
    metric = metric_cls(binder, deck_box, tmp_path / "out.parquet")

    metric.accumulate(_run_row())
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    row = table.to_pylist()[0]
    assert row["run_id"] == "run1"
    assert row[metric_cls.LABEL_COLUMN] == expected_label


@pytest.mark.parametrize("metric_cls,_expected_label", _CASES)
def test_default_output_path_is_under_metrics_sts_gg(
    metric_cls: type[DeckLabelMetric], _expected_label: object
) -> None:
    assert metric_cls.DEFAULT_OUTPUT_PATH.parent == Path("data/metrics/sts_gg")
    assert metric_cls.DEFAULT_OUTPUT_PATH.suffix == ".parquet"


def test_killed_by_is_a_real_null_on_a_loss(tmp_path: Path) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    deck_box = DeckBox()
    metric = KilledByMetric(binder, deck_box, tmp_path / "out.parquet")

    metric.accumulate(_run_row(win=False, killedBy="ENCOUNTER.THE_KIN_BOSS"))
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.to_pylist()[0]["killed_by"] == "ENCOUNTER.THE_KIN_BOSS"


def test_default_output_paths_are_all_distinct() -> None:
    output_paths = {metric_cls.DEFAULT_OUTPUT_PATH for metric_cls, _ in _CASES}
    assert len(output_paths) == len(_CASES)


def test_character_prediction_falls_back_to_other_for_unknown_character(
    tmp_path: Path,
) -> None:
    binder = _binder_from_rows([{"id": "STRIKE_SILENT", "name": "Strike"}], tmp_path)
    deck_box = DeckBox()
    metric = CharacterPredictionMetric(binder, deck_box, tmp_path / "out.parquet")

    metric.accumulate(_run_row(character="CHARACTER.SOME_FUTURE_ADDITION"))
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.to_pylist()[0]["character"] == "OTHER"


def test_character_prediction_label_values_includes_other() -> None:
    assert "OTHER" in CharacterPredictionMetric.LABEL_VALUES
