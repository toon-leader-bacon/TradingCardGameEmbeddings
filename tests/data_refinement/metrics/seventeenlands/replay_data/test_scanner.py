"""Tests for scanner.py's scan_replay_csv() - the same isolation
contract as draft_data/game_data's own scanner tests.
"""

from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.seventeenlands.replay_data.scanner import (
    scan_replay_csv,
)


class _RecordingMetric:
    """A trivial Metric[dict] that records every row/finalize call it
    sees, for asserting scan_replay_csv()'s driving contract."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.finalized = False

    def accumulate(self, row: dict) -> None:
        self.rows.append(row)

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused.parquet")


class _RaisingAccumulateMetric:
    def __init__(self) -> None:
        self.finalized = False

    def accumulate(self, row: dict) -> None:
        raise ValueError("boom - accumulate")

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused.parquet")


class _RaisingFinalizeMetric:
    def accumulate(self, row: dict) -> None:
        pass

    def finalize(self) -> Path:
        raise ValueError("boom - finalize")


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_every_metric_receives_every_row_one_at_a_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "replay_data.csv"
    _write_csv(csv_path, [{"won": True}, {"won": False}])
    metric = _RecordingMetric()

    scan_replay_csv(csv_path, [metric])

    assert len(metric.rows) == 2
    assert metric.rows[0]["won"] is True
    assert metric.rows[1]["won"] is False


def test_one_metrics_accumulate_failure_is_isolated_from_the_others(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "replay_data.csv"
    _write_csv(csv_path, [{"won": True}])
    raising_metric = _RaisingAccumulateMetric()
    healthy_metric = _RecordingMetric()

    scan_replay_csv(csv_path, [raising_metric, healthy_metric])

    assert len(healthy_metric.rows) == 1
    assert raising_metric.finalized is True


def test_one_metrics_finalize_failure_is_isolated_from_the_others(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "replay_data.csv"
    _write_csv(csv_path, [{"won": True}])
    raising_metric = _RaisingFinalizeMetric()
    healthy_metric = _RecordingMetric()

    scan_replay_csv(csv_path, [raising_metric, healthy_metric])

    assert healthy_metric.finalized is True


def test_every_metric_is_finalized_after_the_full_scan(tmp_path: Path) -> None:
    csv_path = tmp_path / "replay_data.csv"
    _write_csv(csv_path, [{"won": True}])
    first_metric = _RecordingMetric()
    second_metric = _RecordingMetric()

    scan_replay_csv(csv_path, [first_metric, second_metric])

    assert first_metric.finalized is True
    assert second_metric.finalized is True
