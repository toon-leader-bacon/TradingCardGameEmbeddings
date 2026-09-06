import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.sts_gg.run_downloader import STSGGRunDownloader

_LEADERBOARD_URL = "https://example.test/leaderboard?page={page}&limit={limit}"
_RUN_DETAIL_URL = "https://example.test/run/{run_id}"


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(
    raw_data_dir: Path, *, per_page_count: int = 100
) -> STSGGRunDownloader:
    return STSGGRunDownloader(
        _fast_rate_limiter(),
        raw_data_dir,
        _LEADERBOARD_URL,
        _RUN_DETAIL_URL,
        per_page_count,
    )


def _leaderboard_page(ids: list[str], *, page: int, total_pages: int) -> dict:
    return {
        "runs": [{"id": run_id} for run_id in ids],
        "total": total_pages * 50,
        "page": page,
        "totalPages": total_pages,
    }


def _mock_text_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = json.dumps(payload)
    return response


class TestInit:
    def test_defaults_urls_and_output_dir_when_omitted(self) -> None:
        downloader = STSGGRunDownloader(_fast_rate_limiter())

        assert downloader.raw_data_dir == STSGGRunDownloader.DEFAULT_RAW_DATA_DIR
        assert downloader.leaderboard_url == STSGGRunDownloader.DEFAULT_LEADERBOARD_URL
        assert downloader.run_detail_url == STSGGRunDownloader.DEFAULT_RUN_DETAIL_URL

    def test_honors_explicit_overrides(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        assert downloader.raw_data_dir == tmp_path
        assert downloader.leaderboard_url == _LEADERBOARD_URL
        assert downloader.run_detail_url == _RUN_DETAIL_URL


class TestPhase1:
    def test_writes_ids_from_a_single_page(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, per_page_count=100)
        page = _mock_text_response(
            _leaderboard_page(["a", "b", "c"], page=1, total_pages=1)
        )

        with patch("requests.get", return_value=page) as mock_get:
            ids_path = downloader.phase_1()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == (
            "https://example.test/leaderboard?page=1&limit=100"
        )
        assert ids_path == tmp_path / "run_ids.txt"
        assert ids_path.read_text(encoding="utf-8") == "a\nb\nc\n"

    def test_pages_through_totalpages_regardless_of_page_size(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        page_1 = _mock_text_response(
            _leaderboard_page(["a", "b"], page=1, total_pages=3)
        )
        page_2 = _mock_text_response(_leaderboard_page(["c"], page=2, total_pages=3))
        page_3 = _mock_text_response(
            _leaderboard_page(["d", "e"], page=3, total_pages=3)
        )

        with patch("requests.get", side_effect=[page_1, page_2, page_3]) as mock_get:
            ids_path = downloader.phase_1()

        assert mock_get.call_count == 3
        assert ids_path.read_text(encoding="utf-8") == "a\nb\nc\nd\ne\n"

    def test_deduplicates_ids_seen_across_pages_preserving_first_seen_order(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        page_1 = _mock_text_response(
            _leaderboard_page(["a", "b"], page=1, total_pages=2)
        )
        page_2 = _mock_text_response(
            _leaderboard_page(["b", "c"], page=2, total_pages=2)
        )

        with patch("requests.get", side_effect=[page_1, page_2]):
            ids_path = downloader.phase_1()

        assert ids_path.read_text(encoding="utf-8") == "a\nb\nc\n"

    def test_raises_when_a_page_fails_after_retries(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch("requests.get", side_effect=requests.ConnectionError("down")):
            with pytest.raises(requests.ConnectionError):
                downloader.phase_1()


class TestPhase2:
    def test_raises_if_phase_1_has_not_run(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(FileNotFoundError):
            downloader.phase_2()

    def test_appends_each_run_not_already_in_manifest(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "run_ids.txt").write_text("run1\nrun2\n", encoding="utf-8")

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            run_id = "run1" if "run1" in url else "run2"
            return _mock_text_response({"id": run_id})

        with patch("requests.get", side_effect=_get_side_effect) as mock_get:
            runs_path = downloader.phase_2()

        assert mock_get.call_count == 2
        assert runs_path == tmp_path / "runs.jsonl"

        rows = [
            json.loads(line)
            for line in runs_path.read_text(encoding="utf-8").splitlines()
        ]
        assert rows == [{"id": "run1"}, {"id": "run2"}]

        manifest_ids = (
            (tmp_path / "runs_manifest.txt").read_text(encoding="utf-8").splitlines()
        )
        assert manifest_ids == ["run1", "run2"]

    def test_skips_a_run_id_already_in_the_manifest(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "run_ids.txt").write_text("run1\nrun2\n", encoding="utf-8")
        (tmp_path / "runs_manifest.txt").write_text("run1\n", encoding="utf-8")
        response = _mock_text_response({"id": "run2"})

        with patch("requests.get", return_value=response) as mock_get:
            downloader.phase_2()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == "https://example.test/run/run2"

    def test_skips_and_logs_a_run_that_fails_after_retries(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "run_ids.txt").write_text("run1\nrun2\n", encoding="utf-8")
        good_response = _mock_text_response({"id": "run2"})

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if "run1" in url:
                raise requests.ConnectionError("down")
            return good_response

        with patch("requests.get", side_effect=_get_side_effect):
            runs_path = downloader.phase_2()

        rows = [
            json.loads(line)
            for line in runs_path.read_text(encoding="utf-8").splitlines()
        ]
        assert rows == [{"id": "run2"}]
        manifest_ids = (
            (tmp_path / "runs_manifest.txt").read_text(encoding="utf-8").splitlines()
        )
        assert manifest_ids == ["run2"]
        assert "run1" in capsys.readouterr().out
