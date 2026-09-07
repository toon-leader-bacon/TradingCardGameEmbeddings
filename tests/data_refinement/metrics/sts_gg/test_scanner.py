import json
import logging
from pathlib import Path

import pytest

from src.data_refinement.metrics.sts_gg.scanner import scan_runs_jsonl


class _RecordingMetric:
    """A fake Metric[dict] that just records what it was fed."""

    def __init__(self, output_path: Path) -> None:
        self.rows: list[dict] = []
        self.finalized = False
        self._output_path = output_path

    def accumulate(self, row: dict) -> None:
        self.rows.append(row)

    def finalize(self) -> Path:
        self.finalized = True
        return self._output_path


class _FailingAccumulateMetric:
    """A fake Metric[dict] whose accumulate() always raises."""

    def accumulate(self, row: dict) -> None:
        raise ValueError("boom")

    def finalize(self) -> Path:
        return Path("unused")


class _FailingFinalizeMetric:
    """A fake Metric[dict] whose finalize() always raises."""

    def accumulate(self, row: dict) -> None:
        pass

    def finalize(self) -> Path:
        raise ValueError("boom")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class TestScanRunsJsonl:
    def test_drives_every_metric_over_every_line(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "runs.jsonl"
        _write_jsonl(raw_path, [{"id": "run1"}, {"id": "run2"}])
        metric_a = _RecordingMetric(tmp_path / "a.parquet")
        metric_b = _RecordingMetric(tmp_path / "b.parquet")

        scan_runs_jsonl(raw_path, [metric_a, metric_b])

        assert [row["id"] for row in metric_a.rows] == ["run1", "run2"]
        assert [row["id"] for row in metric_b.rows] == ["run1", "run2"]
        assert metric_a.finalized
        assert metric_b.finalized

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "runs.jsonl"
        raw_path.write_text(json.dumps({"id": "run1"}) + "\n\n   \n", encoding="utf-8")
        metric = _RecordingMetric(tmp_path / "out.parquet")

        scan_runs_jsonl(raw_path, [metric])

        assert len(metric.rows) == 1

    def test_raises_for_malformed_json_line(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "runs.jsonl"
        raw_path.write_text("{not valid json\n", encoding="utf-8")
        metric = _RecordingMetric(tmp_path / "out.parquet")

        with pytest.raises(json.JSONDecodeError):
            scan_runs_jsonl(raw_path, [metric])

    def test_one_metrics_accumulate_failure_does_not_stop_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        raw_path = tmp_path / "runs.jsonl"
        _write_jsonl(raw_path, [{"id": "run1"}])
        healthy = _RecordingMetric(tmp_path / "out.parquet")
        failing = _FailingAccumulateMetric()

        with caplog.at_level(logging.ERROR):
            scan_runs_jsonl(raw_path, [failing, healthy])

        assert [row["id"] for row in healthy.rows] == ["run1"]
        assert healthy.finalized
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_one_metrics_finalize_failure_does_not_stop_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        raw_path = tmp_path / "runs.jsonl"
        _write_jsonl(raw_path, [{"id": "run1"}])
        healthy = _RecordingMetric(tmp_path / "out.parquet")
        failing = _FailingFinalizeMetric()

        with caplog.at_level(logging.ERROR):
            scan_runs_jsonl(raw_path, [failing, healthy])

        assert healthy.finalized
        assert any(record.levelno == logging.ERROR for record in caplog.records)
