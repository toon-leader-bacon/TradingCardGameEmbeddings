"""Tests for scanner.py's scan_game_csv(): every metric sees every chunk,
every metric is finalized, and one metric's failure is isolated and
logged loudly."""

import logging
from pathlib import Path

import pytest

from src.data_refinement.metrics.isolated_call import FAILURE_MARKER
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import scan_game_csv
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    OWLBEAR,
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

    def accumulate(self, chunk: GameDataChunk) -> None:
        self.chunk_rows.append(len(chunk))

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused.parquet")


class _RaisingAccumulateMetric(_RecordingMetric):
    def accumulate(self, chunk: GameDataChunk) -> None:
        raise ValueError("boom - accumulate")


class _RaisingFinalizeMetric(_RecordingMetric):
    def finalize(self) -> Path:
        raise ValueError("boom - finalize")


def _scan(tmp_path: Path, metrics: list, rows: int = 40, block_size: int = 256) -> None:
    csv_path = write_csv(
        tmp_path / "KTK.TradDraft.csv",
        [row(won=i % 2 == 0, owlbear_deck=1) for i in range(rows)],
    )
    scan_game_csv(
        csv_path, metrics, parser_for(binder_with_cards([OWLBEAR])), block_size
    )


def test_every_metric_sees_every_row_in_chunks(tmp_path: Path) -> None:
    first, second = _RecordingMetric(), _RecordingMetric()

    _scan(tmp_path, [first, second], rows=40)

    assert sum(first.chunk_rows) == 40
    assert len(first.chunk_rows) > 1  # a small block size gives several chunks
    assert first.chunk_rows == second.chunk_rows


def test_every_metric_is_finalized(tmp_path: Path) -> None:
    metrics = [_RecordingMetric(), _RecordingMetric()]

    _scan(tmp_path, metrics)

    assert all(metric.finalized for metric in metrics)


def test_an_accumulate_failure_is_isolated_and_logged_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    healthy, broken = _RecordingMetric(), _RaisingAccumulateMetric()

    with caplog.at_level(logging.ERROR):
        _scan(tmp_path, [broken, healthy])

    assert sum(healthy.chunk_rows) == 40
    assert broken.finalized
    failure_lines = [
        r.getMessage() for r in caplog.records if FAILURE_MARKER in r.getMessage()
    ]
    assert any("KTK.TradDraft.csv chunk 0" in line for line in failure_lines)
    summary = [line for line in failure_lines if "summary" in line]
    assert len(summary) == 1
    chunks = len(healthy.chunk_rows)
    assert f"failed accumulate() on {chunks} of {chunks} chunks" in summary[0]


def test_a_finalize_failure_is_isolated_and_summarized(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    healthy, broken = _RecordingMetric(), _RaisingFinalizeMetric()

    with caplog.at_level(logging.ERROR):
        _scan(tmp_path, [broken, healthy])

    assert healthy.finalized
    summary = [r.getMessage() for r in caplog.records if "summary" in r.getMessage()]
    assert len(summary) == 1
    assert "output missing or untrustworthy" in summary[0]


def test_a_clean_scan_logs_no_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        _scan(tmp_path, [_RecordingMetric()])

    assert not [r for r in caplog.records if FAILURE_MARKER in r.getMessage()]


def test_a_null_scalar_names_the_csv_and_chunk(tmp_path: Path) -> None:
    rows = [row(won=True, owlbear_deck=1) for _ in range(3)]
    rows[2]["won"] = None
    csv_path = write_csv(tmp_path / "KTK.TradDraft.csv", rows)

    with pytest.raises(
        ValueError, match=r"KTK.TradDraft.csv chunk 0: .*'won' is null at batch row 2"
    ):
        scan_game_csv(
            csv_path, [_RecordingMetric()], parser_for(binder_with_cards([OWLBEAR]))
        )
