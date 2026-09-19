import io
import logging
import tarfile
from pathlib import Path

import pytest

from src.data_refinement.metrics.isotropic.games import scanner
from src.data_refinement.metrics.isotropic.games.game_log_parser import GameLog
from src.data_refinement.metrics.isotropic.games.header_parser import GameHeader
from src.data_refinement.metrics.isotropic.games.scanner import (
    scan_isotropic_game_log_archives,
    scan_isotropic_game_logs_archives,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    build_game_log_html,
    build_header_html,
)


class _RecordingMetric:
    def __init__(self) -> None:
        self.headers: list[GameHeader] = []
        self.finalized = False

    def accumulate(self, header: GameHeader) -> None:
        self.headers.append(header)

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused")


class _FailingAccumulateMetric:
    def accumulate(self, header: GameHeader) -> None:
        raise ValueError("boom")

    def finalize(self) -> Path:
        return Path("unused")


class _FailingFinalizeMetric:
    def accumulate(self, header: GameHeader) -> None:
        pass

    def finalize(self) -> Path:
        raise ValueError("boom")


def _write_game_log_archive(archive_path: Path, member_html: dict[str, str]) -> None:
    with tarfile.open(archive_path, "w:bz2") as archive:
        for member_name, html in member_html.items():
            content = html.encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _sample_html(winner: str) -> str:
    return build_header_html(
        winner,
        "All Provinces are gone.",
        ["Witch"],
        [(winner, 40, 20, "Witch", None)],
    )


class TestScanIsotropicGameLogArchives:
    def test_drives_every_metric_over_every_parseable_member(
        self, tmp_path: Path
    ) -> None:
        archive_path = tmp_path / "20130315.tar.bz2"
        _write_game_log_archive(
            archive_path,
            {
                "game-1.html": _sample_html("a"),
                "game-2.html": _sample_html("b"),
            },
        )
        metric_a = _RecordingMetric()
        metric_b = _RecordingMetric()

        scan_isotropic_game_log_archives([archive_path], [metric_a, metric_b])

        assert [h.winner_nick for h in metric_a.headers] == ["a", "b"]
        assert [h.winner_nick for h in metric_b.headers] == ["a", "b"]
        assert metric_a.finalized
        assert metric_b.finalized

    def test_skips_unparseable_member_without_stopping_the_scan(
        self, tmp_path: Path
    ) -> None:
        archive_path = tmp_path / "20130315.tar.bz2"
        _write_game_log_archive(
            archive_path,
            {
                "game-1.html": "<html><body>no pre here</body></html>",
                "game-2.html": _sample_html("b"),
            },
        )
        metric = _RecordingMetric()

        scan_isotropic_game_log_archives([archive_path], [metric])

        assert [h.winner_nick for h in metric.headers] == ["b"]

    def test_pools_multiple_archives(self, tmp_path: Path) -> None:
        archive_1 = tmp_path / "a.tar.bz2"
        archive_2 = tmp_path / "b.tar.bz2"
        _write_game_log_archive(archive_1, {"game-1.html": _sample_html("a")})
        _write_game_log_archive(archive_2, {"game-2.html": _sample_html("b")})
        metric = _RecordingMetric()

        scan_isotropic_game_log_archives([archive_1, archive_2], [metric])

        assert [h.winner_nick for h in metric.headers] == ["a", "b"]

    def test_one_metrics_accumulate_failure_does_not_stop_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive_path = tmp_path / "20130315.tar.bz2"
        _write_game_log_archive(archive_path, {"game-1.html": _sample_html("a")})
        healthy = _RecordingMetric()
        failing = _FailingAccumulateMetric()

        with caplog.at_level(logging.ERROR):
            scan_isotropic_game_log_archives([archive_path], [failing, healthy])

        assert [h.winner_nick for h in healthy.headers] == ["a"]
        assert healthy.finalized
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_one_metrics_finalize_failure_does_not_stop_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive_path = tmp_path / "20130315.tar.bz2"
        _write_game_log_archive(archive_path, {"game-1.html": _sample_html("a")})
        healthy = _RecordingMetric()
        failing = _FailingFinalizeMetric()

        with caplog.at_level(logging.ERROR):
            scan_isotropic_game_log_archives([archive_path], [failing, healthy])

        assert healthy.finalized
        assert any(record.levelno == logging.ERROR for record in caplog.records)


class _RecordingLogMetric:
    def __init__(self) -> None:
        self.game_logs: list[GameLog] = []
        self.finalized = False

    def accumulate(self, game_log: GameLog) -> None:
        self.game_logs.append(game_log)

    def finalize(self) -> Path:
        self.finalized = True
        return Path("unused")


class _FailingLogMetric:
    def accumulate(self, game_log: GameLog) -> None:
        raise ValueError("boom")

    def finalize(self) -> Path:
        return Path("unused")


def _sample_log_html(winner: str) -> str:
    return build_game_log_html(
        ["— a's turn 1 —", "a buys a Witch."],
        winner_nick=winner,
        players=[("a", 40, 1, "Witch", None), ("b", 30, 1, "Silver", None)],
    )


class TestScanIsotropicGameLogsArchives:
    def test_drives_every_metric_over_every_parseable_game_log(
        self, tmp_path: Path
    ) -> None:
        archive_path = tmp_path / "20130315.tar.bz2"
        _write_game_log_archive(
            archive_path,
            {
                "game-1.html": _sample_log_html("a"),
                "game-2.html": "<html><body>no pre here</body></html>",
                "game-3.html": _sample_log_html("b"),
            },
        )
        metric_a = _RecordingLogMetric()
        metric_b = _RecordingLogMetric()

        scan_isotropic_game_logs_archives([archive_path], [metric_a, metric_b])

        assert [g.header.winner_nick for g in metric_a.game_logs] == ["a", "b"]
        assert len(metric_b.game_logs) == 2
        assert metric_a.game_logs[0].turns[0].player_nick == "a"
        assert metric_a.finalized and metric_b.finalized

    def test_parse_exception_skips_only_that_member(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive_path = tmp_path / "20130315.tar.bz2"
        _write_game_log_archive(
            archive_path,
            {"game-1.html": "BOOM", "game-2.html": _sample_log_html("a")},
        )
        real_parse = scanner.parse_game_log

        def _flaky_parse(html_text: str) -> GameLog | None:
            if html_text == "BOOM":
                raise ValueError("boom")
            return real_parse(html_text)

        monkeypatch.setattr(scanner, "parse_game_log", _flaky_parse)
        metric = _RecordingLogMetric()

        with caplog.at_level(logging.ERROR):
            scan_isotropic_game_logs_archives([archive_path], [metric])

        assert len(metric.game_logs) == 1
        assert "game-1.html" in caplog.text

    def test_one_metrics_failure_does_not_stop_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive_path = tmp_path / "20130315.tar.bz2"
        _write_game_log_archive(archive_path, {"game-1.html": _sample_log_html("a")})
        healthy = _RecordingLogMetric()

        with caplog.at_level(logging.ERROR):
            scan_isotropic_game_logs_archives(
                [archive_path], [_FailingLogMetric(), healthy]
            )

        assert len(healthy.game_logs) == 1
        assert healthy.finalized
