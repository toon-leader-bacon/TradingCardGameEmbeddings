"""Tests for scanner.py's scan_replay_csv(): every metric sees every
chunk, every metric is finalized, and one metric's failure is isolated
and logged loudly (the shared chunk_scanner contract)."""

import logging
from pathlib import Path

import pytest

from src.data_refinement.metrics.isolated_call import FAILURE_MARKER
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    ReplayDataChunk,
)
from src.data_refinement.metrics.seventeenlands.replay_data.scanner import (
    scan_replay_csv,
)
from tests.data_refinement.metrics.seventeenlands.replay_data._replay_fixtures import (
    binder_with_cards,
    parser_for,
    row,
    write_csv,
)


class _RecordingMetric:
    """Records the row count of every chunk it sees, and its finalize."""

    def __init__(self) -> None:
        self.chunk_rows: list[int] = []
        self.finalized = False

    def accumulate(self, chunk: ReplayDataChunk) -> None:
        self.chunk_rows.append(len(chunk))

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused.parquet")


class _RaisingAccumulateMetric(_RecordingMetric):
    def accumulate(self, chunk: ReplayDataChunk) -> None:
        raise ValueError("boom - accumulate")


def _scan(tmp_path: Path, metrics: list, rows: int = 40) -> None:
    csv_path = write_csv(
        tmp_path / "PIO.TradSealed.csv",
        [row(owlbear_deck=1, game_number=i) for i in range(rows)],
    )
    scan_replay_csv(csv_path, metrics, parser_for(binder_with_cards()), 2048)


def test_every_metric_sees_every_row_in_chunks(tmp_path: Path) -> None:
    first, second = _RecordingMetric(), _RecordingMetric()

    _scan(tmp_path, [first, second])

    assert sum(first.chunk_rows) == 40
    assert len(first.chunk_rows) > 1  # a small block size gives several chunks
    assert first.chunk_rows == second.chunk_rows
    assert first.finalized and second.finalized


def test_an_accumulate_failure_is_isolated_and_logged_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    healthy, broken = _RecordingMetric(), _RaisingAccumulateMetric()

    with caplog.at_level(logging.ERROR):
        _scan(tmp_path, [broken, healthy])

    assert sum(healthy.chunk_rows) == 40
    assert broken.finalized
    assert any(
        FAILURE_MARKER in r.getMessage()
        and "PIO.TradSealed.csv chunk 0" in r.getMessage()
        for r in caplog.records
    )
