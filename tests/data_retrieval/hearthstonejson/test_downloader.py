from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.hearthstonejson.downloader import (
    BuildDownloadFailure,
    BuildDownloadSuccess,
    HearthstoneJsonDownloader,
)
from src.data_retrieval.rate_limiter import RateLimiter

_LISTING_URL = "https://api.hearthstonejson.com/v1/"

# A representative excerpt of the `tree`-generated root listing page,
# using RELATIVE hrefs — confirmed against the live page (curl'd
# directly, no browser resolution) to be the actual format served,
# e.g. href="/v1/190920/enUS/" rather than an absolute URL.
_EXAMPLE_LISTING_HTML = """
<html><body>
<a href="/v1">/v1</a>
├── <a href="/v1/190920/">190920</a>
│   ├── <a href="/v1/190920/deDE/">deDE</a>
│   ├── <a href="/v1/190920/enUS/">enUS</a>
│   └── <a href="/v1/190920/CardDefs.xml">CardDefs.xml</a>
├── <a href="/v1/191554/">191554</a>
│   ├── <a href="/v1/191554/enUS/">enUS</a>
│   └── <a href="/v1/191554/esES/">esES</a>
├── <a href="/v1/1/">1</a>
│   └── <a href="/v1/1/frFR/">frFR</a>
</body></html>
"""
_EXAMPLE_LISTING_BUILD_IDS_WITH_ENUS = ["190920", "191554"]


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(raw_data_dir: Path) -> HearthstoneJsonDownloader:
    return HearthstoneJsonDownloader(
        listing_url=_LISTING_URL,
        raw_data_dir=raw_data_dir,
        rate_limiter=_fast_rate_limiter(),
    )


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = text
    return response


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class TestParseBuildIdsWithEnusSubdirectory:
    def test_parses_example_listing_page(self) -> None:
        downloader = _make_downloader(Path("unused"))

        build_ids = downloader._parse_build_ids_with_enus_subdirectory(
            _EXAMPLE_LISTING_HTML
        )

        assert build_ids == sorted(_EXAMPLE_LISTING_BUILD_IDS_WITH_ENUS)
        # 190920's deDE/CardDefs.xml links and 1's frFR-only link are
        # correctly excluded — only builds with an enUS link match.
        assert "1" not in build_ids

    def test_deduplicates_repeated_enus_links(self) -> None:
        downloader = _make_downloader(Path("unused"))
        listing_html = (
            '<a href="/v1/190920/enUS/">enUS</a>'
            '<a href="/v1/190920/enUS/">enUS (mirror)</a>'
        )

        build_ids = downloader._parse_build_ids_with_enus_subdirectory(listing_html)

        assert build_ids == ["190920"]

    def test_matches_absolute_url_form_too(self) -> None:
        # Defense in depth: the live page serves relative hrefs today,
        # but tolerate an absolute-URL form as well in case that changes.
        downloader = _make_downloader(Path("unused"))
        listing_html = (
            '<a href="https://api.hearthstonejson.com/v1/190920/enUS/">enUS</a>'
        )

        build_ids = downloader._parse_build_ids_with_enus_subdirectory(listing_html)

        assert build_ids == ["190920"]

    def test_returns_empty_list_for_html_with_no_matches(self) -> None:
        downloader = _make_downloader(Path("unused"))

        build_ids = downloader._parse_build_ids_with_enus_subdirectory(
            "<html><body>nothing here</body></html>"
        )

        assert build_ids == []

    def test_returns_empty_list_for_malformed_input(self) -> None:
        downloader = _make_downloader(Path("unused"))

        build_ids = downloader._parse_build_ids_with_enus_subdirectory("<<<not html>>>")

        assert build_ids == []

    def test_ignores_links_without_enus_subdirectory(self) -> None:
        downloader = _make_downloader(Path("unused"))
        listing_html = (
            '<a href="/v1/190920/deDE/">deDE</a>'
            '<a href="/v1/190920/CardDefs.xml">xml</a>'
        )

        build_ids = downloader._parse_build_ids_with_enus_subdirectory(listing_html)

        assert build_ids == []


class TestListBuildIds:
    def test_fetches_and_parses_listing_page(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response(_EXAMPLE_LISTING_HTML)

        with patch("requests.get", return_value=response) as mock_get:
            build_ids = downloader.list_build_ids()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == _LISTING_URL
        assert build_ids == sorted(_EXAMPLE_LISTING_BUILD_IDS_WITH_ENUS)

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response("")
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                downloader.list_build_ids()


class TestDownloadBuild:
    def test_writes_streamed_response_to_raw_data_dir(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response(
            [b"cards-json-bytes-one-", b"cards-json-bytes-two"]
        )

        with patch("requests.get", return_value=response) as mock_get:
            result_path = downloader.download_build("190920")

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == (
            "https://api.hearthstonejson.com/v1/190920/enUS/cards.json"
        )
        assert result_path == tmp_path / "190920.json"
        assert result_path.read_bytes() == b"cards-json-bytes-one-cards-json-bytes-two"

    def test_creates_raw_data_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response):
            downloader.download_build("190920")

        assert nested_dir.is_dir()

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                downloader.download_build("190920")

    def test_removes_partial_file_on_write_failure(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([b"first-chunk"])
        response.iter_content.side_effect = requests.ConnectionError("dropped")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.ConnectionError):
                downloader.download_build("190920")

        assert not (tmp_path / "190920.json").exists()


class TestDownloadMissingBuilds:
    def test_downloads_only_missing_build_ids(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "111111.json").write_bytes(b"already-here")
        response = _mock_streaming_response([b"newly-downloaded"])

        with patch("requests.get", return_value=response) as mock_get:
            outcomes = downloader.download_missing_builds(["111111", "222222"])

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == (
            "https://api.hearthstonejson.com/v1/222222/enUS/cards.json"
        )
        assert outcomes == [
            BuildDownloadSuccess(build_id="111111", path=tmp_path / "111111.json"),
            BuildDownloadSuccess(build_id="222222", path=tmp_path / "222222.json"),
        ]
        assert (tmp_path / "111111.json").read_bytes() == b"already-here"
        assert (tmp_path / "222222.json").read_bytes() == b"newly-downloaded"

    def test_empty_build_ids_downloads_nothing(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch("requests.get") as mock_get:
            outcomes = downloader.download_missing_builds([])

        mock_get.assert_not_called()
        assert outcomes == []

    def test_failed_build_is_reported_and_loop_continues(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        failing_response = _mock_streaming_response([])
        failing_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Not Found"
        )
        succeeding_response = _mock_streaming_response([b"data"])

        with patch(
            "requests.get", side_effect=[failing_response, succeeding_response]
        ) as mock_get:
            with patch(
                "src.data_retrieval.hearthstonejson.downloader.tqdm.write"
            ) as mock_write:
                outcomes = downloader.download_missing_builds(["111111", "222222"])

        assert mock_get.call_count == 2
        assert outcomes == [
            BuildDownloadFailure(build_id="111111", error="404 Not Found"),
            BuildDownloadSuccess(build_id="222222", path=tmp_path / "222222.json"),
        ]
        # The failing build didn't stop 222222 from being attempted and
        # succeeding, and the failure was reported via tqdm.write (not
        # print), so it doesn't corrupt the progress bar.
        assert not (tmp_path / "111111.json").exists()
        assert (tmp_path / "222222.json").read_bytes() == b"data"
        mock_write.assert_called_once()
        written_message = mock_write.call_args.args[0]
        assert "111111" in written_message
        assert "404 Not Found" in written_message

    def test_all_builds_failing_returns_all_failures(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            outcomes = downloader.download_missing_builds(["111111", "222222"])

        assert outcomes == [
            BuildDownloadFailure(build_id="111111", error="404 Not Found"),
            BuildDownloadFailure(build_id="222222", error="404 Not Found"),
        ]

    def test_success_then_failure_preserves_order(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        succeeding_response = _mock_streaming_response([b"data"])
        failing_response = _mock_streaming_response([])
        failing_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Not Found"
        )

        with patch("requests.get", side_effect=[succeeding_response, failing_response]):
            outcomes = downloader.download_missing_builds(["111111", "222222"])

        assert outcomes == [
            BuildDownloadSuccess(build_id="111111", path=tmp_path / "111111.json"),
            BuildDownloadFailure(build_id="222222", error="404 Not Found"),
        ]


class TestFetch:
    def test_composes_list_build_ids_and_download_missing_builds(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        expected_outcomes = [
            BuildDownloadSuccess(build_id="190920", path=tmp_path / "190920.json")
        ]

        with patch.object(
            downloader, "list_build_ids", return_value=["190920"]
        ) as mock_list:
            with patch.object(
                downloader, "download_missing_builds", return_value=expected_outcomes
            ) as mock_download:
                result = downloader.fetch()

        mock_list.assert_called_once_with()
        mock_download.assert_called_once_with(["190920"])
        assert result == expected_outcomes
