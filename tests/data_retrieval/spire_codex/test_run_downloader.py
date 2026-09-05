import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.spire_codex.run_downloader import (
    SpireCodexRunDownloader,
    _ExportComplete,
    _FetchedPage,
    _PagesRemaining,
)

_EXPORT_URL = "https://example.test/api/exports/runs"
_STATS_URL = "https://example.test/api/runs/stats"


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(raw_data_dir: Path, *, page_limit: int | None = None) -> SpireCodexRunDownloader:
    return SpireCodexRunDownloader(
        export_url=_EXPORT_URL,
        raw_data_dir=raw_data_dir,
        stats_url=_STATS_URL,
        page_limit=page_limit,
        rate_limiter=_fast_rate_limiter(),
    )


def _mock_page_response(chunks: list[bytes], next_cursor: str | None) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.headers = {"X-Next-Cursor": next_cursor} if next_cursor is not None else {}
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def _mock_stats_response(total_runs: int) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"total_runs": total_runs}
    return response


def _write_page(raw_data_dir: Path, page_index: int, cursor: str | None) -> None:
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    (raw_data_dir / f"page_{page_index:05d}.jsonl.gz").write_bytes(b"gzip-bytes")
    (raw_data_dir / f"page_{page_index:05d}.next_cursor").write_text(
        cursor or "", encoding="utf-8"
    )


class TestScanDownloadedPages:
    def test_returns_empty_list_for_empty_directory(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        assert downloader._scan_downloaded_pages() == []

    def test_reads_complete_pages_in_order(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        _write_page(tmp_path, 0, "cursor-1")
        _write_page(tmp_path, 1, None)

        pages = downloader._scan_downloaded_pages()

        assert pages == [
            _FetchedPage(path=tmp_path / "page_00000.jsonl.gz", next_cursor="cursor-1"),
            _FetchedPage(path=tmp_path / "page_00001.jsonl.gz", next_cursor=None),
        ]

    def test_stops_at_first_gap(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        _write_page(tmp_path, 0, "cursor-1")
        # page 1 missing its sidecar — incomplete, so not "downloaded".
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "page_00001.jsonl.gz").write_bytes(b"partial")
        _write_page(tmp_path, 2, None)

        pages = downloader._scan_downloaded_pages()

        assert pages == [
            _FetchedPage(path=tmp_path / "page_00000.jsonl.gz", next_cursor="cursor-1")
        ]


class TestResumePoint:
    def test_fresh_walk_starts_at_page_zero(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        assert downloader._resume_point([]) == _PagesRemaining(next_page_index=0, cursor=None)

    def test_continues_from_last_page_with_a_cursor(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        downloaded_pages = [
            _FetchedPage(path=tmp_path / "page_00000.jsonl.gz", next_cursor="cursor-1")
        ]

        assert downloader._resume_point(downloaded_pages) == _PagesRemaining(
            next_page_index=1, cursor="cursor-1"
        )

    def test_export_complete_when_last_page_has_no_next_cursor(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        downloaded_pages = [
            _FetchedPage(path=tmp_path / "page_00000.jsonl.gz", next_cursor=None)
        ]

        assert downloader._resume_point(downloaded_pages) == _ExportComplete()


class TestRecordWindow:
    def test_writes_window_metadata_on_first_call(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)

        downloader._record_window("2026-01-01", "2026-02-01")

        window_path = tmp_path / "window.json"
        assert json.loads(window_path.read_text(encoding="utf-8")) == {
            "start": "2026-01-01",
            "end": "2026-02-01",
        }

    def test_matching_window_is_a_no_op(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        downloader._record_window("2026-01-01", None)

        downloader._record_window("2026-01-01", None)  # should not raise

    def test_mismatched_window_raises(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        downloader._record_window("2026-01-01", None)

        with pytest.raises(ValueError, match="does not match"):
            downloader._record_window("2026-02-01", None)


class TestEstimateTotalPages:
    def test_computes_ceiling_of_total_runs_over_page_limit(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, page_limit=50000)
        response = _mock_stats_response(total_runs=1_527_664)

        with patch("requests.get", return_value=response) as mock_get:
            estimate = downloader._estimate_total_pages()

        mock_get.assert_called_once_with(_STATS_URL, timeout=60)
        assert estimate == 31  # ceil(1_527_664 / 50_000)

    def test_returns_none_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_stats_response(total_runs=0)
        response.raise_for_status.side_effect = requests.HTTPError("500")

        with patch("requests.get", return_value=response):
            assert downloader._estimate_total_pages() is None

    def test_returns_none_on_missing_total_runs_field(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"unexpected": "shape"}

        with patch("requests.get", return_value=response):
            assert downloader._estimate_total_pages() is None


class TestFetchPage:
    def test_writes_page_and_cursor_sidecar(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        response = _mock_page_response([b"gz-bytes-a", b"gz-bytes-b"], next_cursor="cursor-1")

        with patch("requests.get", return_value=response) as mock_get:
            page = downloader._fetch_page(0, None, start=None, end=None)

        assert mock_get.call_args.args[0] == _EXPORT_URL
        assert mock_get.call_args.kwargs["params"] == {"limit": downloader.page_limit}
        assert page.path == tmp_path / "page_00000.jsonl.gz"
        assert page.next_cursor == "cursor-1"
        assert page.path.read_bytes() == b"gz-bytes-agz-bytes-b"
        assert (tmp_path / "page_00000.next_cursor").read_text(encoding="utf-8") == "cursor-1"

    def test_empty_sidecar_when_no_next_cursor_header(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        response = _mock_page_response([b"gz-bytes"], next_cursor=None)

        with patch("requests.get", return_value=response):
            page = downloader._fetch_page(0, None, start=None, end=None)

        assert page.next_cursor is None
        assert (tmp_path / "page_00000.next_cursor").read_text(encoding="utf-8") == ""

    def test_passes_cursor_start_end_as_params(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        response = _mock_page_response([b"data"], next_cursor=None)

        with patch("requests.get", return_value=response) as mock_get:
            downloader._fetch_page(
                3, "prior-cursor", start="2026-01-01", end="2026-02-01"
            )

        assert mock_get.call_args.kwargs["params"] == {
            "limit": downloader.page_limit,
            "cursor": "prior-cursor",
            "start": "2026-01-01",
            "end": "2026-02-01",
        }

    def test_removes_partial_page_file_on_write_failure(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        response = _mock_page_response([b"first-chunk"], next_cursor="cursor-1")
        response.iter_content.side_effect = requests.ConnectionError("dropped")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.ConnectionError):
                downloader._fetch_page(0, None, start=None, end=None)

        assert not (tmp_path / "page_00000.jsonl.gz").exists()
        assert not (tmp_path / "page_00000.next_cursor").exists()

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        response = _mock_page_response([], next_cursor=None)
        response.raise_for_status.side_effect = requests.HTTPError("404")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                downloader._fetch_page(0, None, start=None, end=None)


class TestFetch:
    def test_creates_raw_data_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        responses = [_mock_stats_response(total_runs=1), _mock_page_response([b"x"], None)]

        with patch("requests.get", side_effect=responses):
            downloader.fetch()

        assert nested_dir.is_dir()

    def test_walks_every_page_until_no_next_cursor(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        responses = [
            _mock_stats_response(total_runs=100_000),
            _mock_page_response([b"page-0-bytes"], next_cursor="cursor-1"),
            _mock_page_response([b"page-1-bytes"], next_cursor=None),
        ]

        with patch("requests.get", side_effect=responses) as mock_get:
            page_paths = downloader.fetch()

        assert mock_get.call_count == 3  # stats + 2 pages
        assert page_paths == [
            tmp_path / "page_00000.jsonl.gz",
            tmp_path / "page_00001.jsonl.gz",
        ]
        assert (tmp_path / "page_00000.jsonl.gz").read_bytes() == b"page-0-bytes"
        assert (tmp_path / "page_00001.jsonl.gz").read_bytes() == b"page-1-bytes"
        assert json.loads((tmp_path / "window.json").read_text(encoding="utf-8")) == {
            "start": None,
            "end": None,
        }

    def test_resumes_from_last_complete_page(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        downloader._record_window(None, None)
        _write_page(tmp_path, 0, "cursor-1")
        responses = [
            _mock_stats_response(total_runs=100_000),
            _mock_page_response([b"page-1-bytes"], next_cursor=None),
        ]

        with patch("requests.get", side_effect=responses) as mock_get:
            page_paths = downloader.fetch()

        assert mock_get.call_count == 2  # stats + only the missing page
        assert page_paths == [
            tmp_path / "page_00000.jsonl.gz",
            tmp_path / "page_00001.jsonl.gz",
        ]

    def test_already_complete_export_makes_no_page_requests(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        downloader._record_window(None, None)
        _write_page(tmp_path, 0, None)

        with patch("requests.get") as mock_get:
            page_paths = downloader.fetch()

        mock_get.assert_not_called()
        assert page_paths == [tmp_path / "page_00000.jsonl.gz"]

    def test_raises_on_window_mismatch_before_any_request(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        downloader._record_window("2026-01-01", None)

        with patch("requests.get") as mock_get:
            with pytest.raises(ValueError, match="does not match"):
                downloader.fetch(start="2026-02-01")

        mock_get.assert_not_called()

    def test_succeeds_when_stats_estimate_call_fails(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        failing_stats_response = _mock_stats_response(total_runs=0)
        failing_stats_response.raise_for_status.side_effect = requests.HTTPError("500")
        responses = [failing_stats_response, _mock_page_response([b"page-0"], None)]

        with patch("requests.get", side_effect=responses):
            page_paths = downloader.fetch()

        assert page_paths == [tmp_path / "page_00000.jsonl.gz"]
