import io
import json
import logging
import tarfile
from pathlib import Path

import pytest

from src.data_refinement.metrics.isotropic.summary.scanner import (
    scan_isotropic_summary_archives,
)


class _RecordingMetric:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.finalized = False

    def accumulate(self, row: dict) -> None:
        self.rows.append(row)

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused")


class _FailingAccumulateMetric:
    def accumulate(self, row: dict) -> None:
        raise ValueError("boom")

    def finalize(self) -> Path:
        return Path("unused")


class _FailingFinalizeMetric:
    def accumulate(self, row: dict) -> None:
        pass

    def finalize(self) -> Path:
        raise ValueError("boom")


def _write_summary_archive(archive_path: Path, members: dict[str, list[dict]]) -> None:
    """Build a *-summary.tar.bz2 with one games-*.json member per key."""
    with tarfile.open(archive_path, "w:bz2") as archive:
        for member_name, rows in members.items():
            content = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(content)

            archive.addfile(info, io.BytesIO(content))


class TestScanIsotropicSummaryArchives:
    def test_drives_every_metric_over_every_row_across_members(
        self, tmp_path: Path
    ) -> None:
        archive_path = tmp_path / "2020_202001-summary.tar.bz2"
        _write_summary_archive(
            archive_path,
            {
                "games-20200101.json": [{"serial": 1}, {"serial": 2}],
                "games-20200102.json": [{"serial": 3}],
            },
        )
        metric_a = _RecordingMetric()
        metric_b = _RecordingMetric()

        scan_isotropic_summary_archives([archive_path], [metric_a, metric_b])

        assert [row["serial"] for row in metric_a.rows] == [1, 2, 3]
        assert [row["serial"] for row in metric_b.rows] == [1, 2, 3]
        assert metric_a.finalized
        assert metric_b.finalized

    def test_pools_multiple_archives(self, tmp_path: Path) -> None:
        archive_1 = tmp_path / "a-summary.tar.bz2"
        archive_2 = tmp_path / "b-summary.tar.bz2"
        _write_summary_archive(archive_1, {"games-20100101.json": [{"serial": 1}]})
        _write_summary_archive(archive_2, {"games-20130101.json": [{"serial": 2}]})
        metric = _RecordingMetric()

        scan_isotropic_summary_archives([archive_1, archive_2], [metric])

        assert [row["serial"] for row in metric.rows] == [1, 2]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "summary.tar.bz2"
        with tarfile.open(archive_path, "w:bz2") as archive:

            content = (json.dumps({"serial": 1}) + "\n\n   \n").encode("utf-8")
            info = tarfile.TarInfo(name="games-20200101.json")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        metric = _RecordingMetric()

        scan_isotropic_summary_archives([archive_path], [metric])

        assert len(metric.rows) == 1

    def test_one_metrics_accumulate_failure_does_not_stop_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive_path = tmp_path / "summary.tar.bz2"
        _write_summary_archive(archive_path, {"games-20200101.json": [{"serial": 1}]})
        healthy = _RecordingMetric()
        failing = _FailingAccumulateMetric()

        with caplog.at_level(logging.ERROR):
            scan_isotropic_summary_archives([archive_path], [failing, healthy])

        assert [row["serial"] for row in healthy.rows] == [1]
        assert healthy.finalized
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_one_metrics_finalize_failure_does_not_stop_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive_path = tmp_path / "summary.tar.bz2"
        _write_summary_archive(archive_path, {"games-20200101.json": [{"serial": 1}]})
        healthy = _RecordingMetric()
        failing = _FailingFinalizeMetric()

        with caplog.at_level(logging.ERROR):
            scan_isotropic_summary_archives([archive_path], [failing, healthy])

        assert healthy.finalized
        assert any(record.levelno == logging.ERROR for record in caplog.records)
