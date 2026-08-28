from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from src.data_refinement.seventeenlands.metric_checkpoint import MetricCheckpoint
from src.data_refinement.seventeenlands.metric_result import MetricResult
from src.data_refinement.seventeenlands.metric_scan_loop import accumulate_over_chunks


class _FakeMetric:
    name = "fake_metric"

    def __init__(self) -> None:
        self.rows_seen = 0
        self.chunks_seen = 0
        self.resolved_values_seen: list[object] = []

    def accumulate(self, chunk: pd.DataFrame, resolved: object) -> None:
        self.rows_seen += len(chunk)
        self.chunks_seen += 1
        self.resolved_values_seen.append(resolved)

    def finalize(self) -> dict[UUID, MetricResult]:
        return {}

    def save_state(self) -> dict:
        return {"rows_seen": self.rows_seen, "chunks_seen": self.chunks_seen}

    def load_state(self, state: dict) -> None:
        self.rows_seen = state["rows_seen"]
        self.chunks_seen = state["chunks_seen"]


class _RaisingAfterNChunksMetric(_FakeMetric):
    def __init__(self, raise_after_chunks: int) -> None:
        super().__init__()
        self._raise_after_chunks = raise_after_chunks

    def accumulate(self, chunk: pd.DataFrame, resolved: object) -> None:
        if self.chunks_seen >= self._raise_after_chunks:
            raise RuntimeError("simulated crash")
        super().accumulate(chunk, resolved)


def _write_csv(path: Path, num_rows: int) -> None:
    lines = ["value\n"]
    for i in range(num_rows):
        lines.append(f"{i}\n")
    path.write_text("".join(lines))


class TestAccumulateOverChunks:
    def test_accumulates_all_rows_and_returns_cumulative_count(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=10)
        metric = _FakeMetric()
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        rows_processed, extra = accumulate_over_chunks(
            csv_path,
            [metric],
            lambda chunk: "resolved",
            checkpoint,
            0,
            10,
            "test scan",
        )

        assert rows_processed == 10
        assert metric.rows_seen == 10
        assert extra == {}

    def test_resolve_callback_receives_each_chunk(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=5)
        metric = _FakeMetric()
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        accumulate_over_chunks(
            csv_path,
            [metric],
            lambda chunk: len(chunk),
            checkpoint,
            0,
            10,
            "test scan",
        )

        assert metric.resolved_values_seen == [5]

    def test_no_active_metrics_does_not_raise(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=3)
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        rows_processed, extra = accumulate_over_chunks(
            csv_path,
            [],
            lambda chunk: None,
            checkpoint,
            0,
            10,
            "test scan",
        )

        assert rows_processed == 3
        assert extra == {}

    def test_resumes_from_rows_already_processed(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=10)
        metric = _FakeMetric()
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        rows_processed, _ = accumulate_over_chunks(
            csv_path,
            [metric],
            lambda chunk: None,
            checkpoint,
            4,
            10,
            "test scan",
        )

        # Started at row offset 4 of a 10-row file -> only 6 new rows seen,
        # but the returned count is cumulative (4 + 6 = 10).
        assert metric.rows_seen == 6
        assert rows_processed == 10

    def test_checkpoint_extra_computes_final_return_value(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=3)
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        _, extra = accumulate_over_chunks(
            csv_path,
            [_FakeMetric()],
            lambda chunk: None,
            checkpoint,
            0,
            10,
            "test scan",
            checkpoint_extra=lambda: {"marker": "final"},
        )

        assert extra == {"marker": "final"}

    def test_no_checkpoint_extra_returns_empty_dict(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=3)
        checkpoint = MetricCheckpoint(tmp_path / "checkpoint.json")

        _, extra = accumulate_over_chunks(
            csv_path,
            [_FakeMetric()],
            lambda chunk: None,
            checkpoint,
            0,
            10,
            "test scan",
        )

        assert extra == {}

    def test_final_checkpoint_extra_call_does_not_write_to_disk(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=3)
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint = MetricCheckpoint(checkpoint_path)

        # checkpoint_every_n_chunks is larger than the number of chunks
        # this tiny file produces (1), so no PERIODIC write should ever
        # fire — the final checkpoint_extra() call (for the return
        # value) must not itself write to disk.
        accumulate_over_chunks(
            csv_path,
            [_FakeMetric()],
            lambda chunk: None,
            checkpoint,
            0,
            1000,
            "test scan",
            checkpoint_extra=lambda: {"marker": "final"},
        )

        assert not checkpoint_path.exists()

    def test_checkpoint_written_periodically_with_cumulative_rows(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, num_rows=100_001)
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint = MetricCheckpoint(checkpoint_path)
        metric = _RaisingAfterNChunksMetric(raise_after_chunks=1)

        with pytest.raises(RuntimeError):
            accumulate_over_chunks(
                csv_path,
                [metric],
                lambda chunk: None,
                checkpoint,
                0,
                1,
                "test scan",
                checkpoint_extra=lambda: {"marker": "periodic"},
            )

        assert checkpoint_path.exists()
        rows_processed, extra = checkpoint.restore([_FakeMetric()])
        assert rows_processed == 100_000  # only chunk 1 succeeded
        assert extra == {"marker": "periodic"}
