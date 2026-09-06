from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.gwent_one.downloader import GwentOneDownloader
from src.data_retrieval.rate_limiter import RateLimiter

_AJAX_URL = "https://gwent.one/search/ajax"


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(raw_data_dir: Path, pages: int = 1) -> GwentOneDownloader:
    return GwentOneDownloader(
        result_limit=1300,
        raw_data_dir=raw_data_dir,
        rate_limiter=_fast_rate_limiter(),
        pages=pages,
    )


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = text
    return response


class TestInit:
    def test_defaults_ajax_url_and_raw_data_dir_when_omitted(self) -> None:
        downloader = GwentOneDownloader(
            result_limit=1300, rate_limiter=_fast_rate_limiter()
        )

        assert downloader.ajax_url == GwentOneDownloader.DEFAULT_AJAX_URL
        assert downloader.raw_data_dir == GwentOneDownloader.DEFAULT_RAW_DATA_DIR
        assert downloader.language == "en"
        assert downloader.pages == 1

    def test_honors_explicit_overrides(self, tmp_path: Path) -> None:
        downloader = GwentOneDownloader(
            1300,
            rate_limiter=_fast_rate_limiter(),
            raw_data_dir=tmp_path,
            pages=2,
            ajax_url="https://example.test/ajax",
            language="de",
        )

        assert downloader.ajax_url == "https://example.test/ajax"
        assert downloader.raw_data_dir == tmp_path
        assert downloader.language == "de"
        assert downloader.pages == 2


class TestFetchPage:
    def test_posts_expected_form_body_and_saves_response(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response("<div class='card-wrap card-data'></div>")

        with patch("requests.post", return_value=response) as mock_post:
            result_path = downloader.fetch_page(page=1, result_limit=1300)

        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == _AJAX_URL
        assert mock_post.call_args.kwargs["data"] == {
            "total": 1300,
            "lang": "en",
            "page": 1,
        }
        assert result_path == tmp_path / "page_1.html"
        assert result_path.read_text(encoding="utf-8") == (
            "<div class='card-wrap card-data'></div>"
        )

    def test_creates_raw_data_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        response = _mock_text_response("<html></html>")

        with patch("requests.post", return_value=response):
            downloader.fetch_page(page=1, result_limit=1300)

        assert nested_dir.is_dir()

    def test_always_overwrites_existing_page_file(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "page_1.html").write_text("stale", encoding="utf-8")
        response = _mock_text_response("fresh")

        with patch("requests.post", return_value=response):
            downloader.fetch_page(page=1, result_limit=1300)

        assert (tmp_path / "page_1.html").read_text(encoding="utf-8") == "fresh"

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response("")
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with patch("requests.post", return_value=response):
            with pytest.raises(requests.HTTPError):
                downloader.fetch_page(page=1, result_limit=1300)

    def test_removes_partial_file_on_write_failure(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response("full response text")
        destination = tmp_path / "page_1.html"

        def _write_partial_then_fail(self: Path, data: str, encoding: str) -> None:
            # A real partial write, so the file genuinely exists
            # mid-failure — proves the except/unlink cleanup actually
            # ran, rather than trivially passing because the mock
            # never touched the filesystem at all.
            self.write_bytes(data[: len(data) // 2].encode(encoding))
            raise OSError("disk full")

        with patch("requests.post", return_value=response):
            with patch.object(Path, "write_text", _write_partial_then_fail):
                with pytest.raises(OSError):
                    downloader.fetch_page(page=1, result_limit=1300)

        assert not destination.exists()


class TestPhase1:
    def test_composes_fetch_page_per_page_and_returns_raw_data_dir(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path, pages=2)

        with patch.object(
            downloader,
            "fetch_page",
            side_effect=[tmp_path / "page_1.html", tmp_path / "page_2.html"],
        ) as mock_fetch_page:
            result = downloader.phase_1()

        assert mock_fetch_page.call_args_list == [
            ((1, 1300),),
            ((2, 1300),),
        ]
        assert result == tmp_path

    def test_defaults_to_a_single_page(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch.object(
            downloader, "fetch_page", return_value=tmp_path / "page_1.html"
        ) as mock_fetch_page:
            result = downloader.phase_1()

        mock_fetch_page.assert_called_once_with(1, 1300)
        assert result == tmp_path
