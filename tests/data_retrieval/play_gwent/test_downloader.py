import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.play_gwent.downloader import (
    _RETRY_BACKOFF_SECONDS,
    PlayGwentDownloader,
)
from src.data_retrieval.rate_limiter import RateLimiter
from tests.data_retrieval.play_gwent.guide_detail_page_fixture import (
    DOUBLE_QUOTED_GUIDE_PAGE_HTML,
    GUIDE_PAGE_HTML_WITH_NO_GUIDE_KEY,
    GUIDE_PAGE_HTML_WITH_NO_ROOT_DIV,
    GUIDE_PAGE_HTML_WITHOUT_DATA_STATE,
    REAL_GUIDE_CARDS,
    REAL_GUIDE_ID,
    REAL_GUIDE_NAME,
    SINGLE_QUOTED_GUIDE_PAGE_HTML,
)

_DECK_ID_GETTER_URL = "https://example.test/guides/offset/{offset}/limit/{limit}"
_DECK_DETAIL_URL = "https://example.test/guides/{guide_id}"


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(
    raw_data_dir: Path,
    *,
    initial_offset: int = 0,
    per_page_count: int = 500,
) -> PlayGwentDownloader:
    return PlayGwentDownloader(
        _fast_rate_limiter(),
        raw_data_dir,
        _DECK_ID_GETTER_URL,
        _DECK_DETAIL_URL,
        initial_offset,
        per_page_count,
    )


def _mock_json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = text
    return response


def _guides_page(ids: list[int]) -> dict:
    return {"guides": [{"id": guide_id} for guide_id in ids]}


class TestInit:
    def test_defaults_urls_and_output_dir_when_omitted(self) -> None:
        downloader = PlayGwentDownloader(_fast_rate_limiter())

        assert downloader.raw_data_dir == PlayGwentDownloader.DEFAULT_RAW_DATA_DIR
        assert (
            downloader.deck_id_getter_url
            == PlayGwentDownloader.DEFAULT_DECK_ID_GETTER_URL
        )
        assert downloader.deck_detail_url == PlayGwentDownloader.DEFAULT_DECK_DETAIL_URL

    def test_honors_explicit_overrides(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        assert downloader.raw_data_dir == tmp_path
        assert downloader.deck_id_getter_url == _DECK_ID_GETTER_URL
        assert downloader.deck_detail_url == _DECK_DETAIL_URL


class TestPhase1:
    def test_writes_single_page_of_ids(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, per_page_count=500)
        response = _mock_json_response(_guides_page([1, 2, 3]))

        with patch("requests.get", return_value=response) as mock_get:
            ids_path = downloader.phase_1()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == (
            "https://example.test/guides/offset/0/limit/500"
        )
        assert ids_path == tmp_path / "deck_ids.txt"
        assert ids_path.read_text(encoding="utf-8") == "1\n2\n3\n"

    def test_pages_until_a_short_page_is_returned(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, per_page_count=2)
        full_page = _mock_json_response(_guides_page([1, 2]))
        short_page = _mock_json_response(_guides_page([3]))

        with patch("requests.get", side_effect=[full_page, short_page]) as mock_get:
            ids_path = downloader.phase_1()

        assert mock_get.call_args_list[0].args[0] == (
            "https://example.test/guides/offset/0/limit/2"
        )
        assert mock_get.call_args_list[1].args[0] == (
            "https://example.test/guides/offset/2/limit/2"
        )
        assert ids_path.read_text(encoding="utf-8") == "1\n2\n3\n"

    def test_stops_on_empty_page(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, per_page_count=2)
        full_page = _mock_json_response(_guides_page([1, 2]))
        empty_page = _mock_json_response(_guides_page([]))

        with patch("requests.get", side_effect=[full_page, empty_page]) as mock_get:
            ids_path = downloader.phase_1()

        assert mock_get.call_count == 2
        assert ids_path.read_text(encoding="utf-8") == "1\n2\n"

    def test_deduplicates_ids_preserving_first_seen_order(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, per_page_count=3)
        full_page = _mock_json_response(_guides_page([1, 2, 1]))
        short_page = _mock_json_response(_guides_page([2, 3]))

        with patch("requests.get", side_effect=[full_page, short_page]):
            ids_path = downloader.phase_1()

        assert ids_path.read_text(encoding="utf-8") == "1\n2\n3\n"

    def test_starts_at_initial_offset(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, initial_offset=1000, per_page_count=500)
        response = _mock_json_response(_guides_page([1]))

        with patch("requests.get", return_value=response) as mock_get:
            downloader.phase_1()

        assert mock_get.call_args.args[0] == (
            "https://example.test/guides/offset/1000/limit/500"
        )

    def test_empty_result_writes_empty_file(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, per_page_count=500)
        response = _mock_json_response(_guides_page([]))

        with patch("requests.get", return_value=response):
            ids_path = downloader.phase_1()

        assert ids_path.read_text(encoding="utf-8") == ""

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir, per_page_count=500)
        response = _mock_json_response(_guides_page([1]))

        with patch("requests.get", return_value=response):
            downloader.phase_1()

        assert nested_dir.is_dir()

    def test_raises_after_exhausting_retries_on_http_error(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_json_response({"guides": []})
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with patch("requests.get", return_value=response) as mock_get:
            with patch(
                "src.data_retrieval.play_gwent.downloader.time.sleep"
            ) as mock_sleep:
                with pytest.raises(requests.HTTPError):
                    downloader.phase_1()

        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    def test_raises_when_response_has_no_guides_key(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_json_response({"somethingElse": []})

        with patch("requests.get", return_value=response):
            with pytest.raises(KeyError):
                downloader.phase_1()


class TestExtractGuidePayload:
    def test_extracts_from_double_quoted_data_state(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        guide = downloader._extract_guide_payload(DOUBLE_QUOTED_GUIDE_PAGE_HTML)

        assert guide["id"] == REAL_GUIDE_ID
        assert guide["name"] == REAL_GUIDE_NAME
        assert guide["cards"] == REAL_GUIDE_CARDS

    def test_extracts_from_single_quoted_data_state(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        guide = downloader._extract_guide_payload(SINGLE_QUOTED_GUIDE_PAGE_HTML)

        assert guide["id"] == REAL_GUIDE_ID
        assert guide["cards"] == REAL_GUIDE_CARDS

    def test_raises_when_data_state_attribute_missing(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(ValueError, match="no data-state attribute"):
            downloader._extract_guide_payload(GUIDE_PAGE_HTML_WITHOUT_DATA_STATE)

    def test_raises_when_root_div_missing(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(ValueError, match="No element matching"):
            downloader._extract_guide_payload(GUIDE_PAGE_HTML_WITH_NO_ROOT_DIV)

    def test_ignores_decoy_style_block_and_quoted_attribute(
        self, tmp_path: Path
    ) -> None:
        # Both fixture pages already embed a decoy <style> block full
        # of unquoted `data-state=...` CSS selectors and an unrelated
        # element with its own quoted data-state attribute, ahead of
        # the real div#root in document order — this asserts
        # extraction still finds the right one, not just "a" one.
        downloader = _make_downloader(tmp_path)

        guide = downloader._extract_guide_payload(DOUBLE_QUOTED_GUIDE_PAGE_HTML)

        assert guide["id"] == REAL_GUIDE_ID

    def test_raises_when_guide_key_missing(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(ValueError, match="no 'guide' key"):
            downloader._extract_guide_payload(GUIDE_PAGE_HTML_WITH_NO_GUIDE_KEY)


class TestPhase2:
    def test_raises_if_deck_ids_file_missing(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(FileNotFoundError):
            downloader.phase_2()

    def test_fetches_each_id_and_writes_guides_and_manifest(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text("407608\n999\n", encoding="utf-8")
        response = _mock_text_response(DOUBLE_QUOTED_GUIDE_PAGE_HTML)

        with patch("requests.get", return_value=response) as mock_get:
            guides_path = downloader.phase_2()

        assert mock_get.call_args_list[0].args[0] == (
            "https://example.test/guides/407608"
        )
        assert mock_get.call_args_list[1].args[0] == "https://example.test/guides/999"
        assert guides_path == tmp_path / "guides.jsonl"

        written_lines = guides_path.read_text(encoding="utf-8").splitlines()
        assert len(written_lines) == 2
        assert json.loads(written_lines[0])["id"] == REAL_GUIDE_ID

        manifest_ids = (
            (tmp_path / "guides_manifest.txt").read_text(encoding="utf-8").splitlines()
        )
        assert manifest_ids == ["407608", "999"]

    def test_skips_ids_already_in_manifest(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text("407608\n999\n", encoding="utf-8")
        (tmp_path / "guides_manifest.txt").write_text("407608\n", encoding="utf-8")
        response = _mock_text_response(DOUBLE_QUOTED_GUIDE_PAGE_HTML)

        with patch("requests.get", return_value=response) as mock_get:
            downloader.phase_2()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == "https://example.test/guides/999"

    def test_appends_data_row_before_manifest_row_on_each_guide(
        self, tmp_path: Path
    ) -> None:
        # Regression guard for the crash-safety write ordering: a
        # partial run (data row written, manifest not yet updated)
        # must look like "not yet downloaded", not "corrupted".
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text("407608\n", encoding="utf-8")
        response = _mock_text_response(DOUBLE_QUOTED_GUIDE_PAGE_HTML)

        real_open = open
        write_order: list[str] = []

        def _tracking_open(path, mode="r", *args, **kwargs):
            if mode == "a":
                write_order.append(Path(path).name)
            return real_open(path, mode, *args, **kwargs)

        with patch("requests.get", return_value=response):
            with patch(
                "src.data_retrieval.download_utils.open",
                side_effect=_tracking_open,
                create=True,
            ):
                downloader.phase_2()

        assert write_order == ["guides.jsonl", "guides_manifest.txt"]

    def test_all_ids_already_downloaded_makes_no_requests(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text("407608\n", encoding="utf-8")
        (tmp_path / "guides_manifest.txt").write_text("407608\n", encoding="utf-8")

        with patch("requests.get") as mock_get:
            guides_path = downloader.phase_2()

        mock_get.assert_not_called()
        assert guides_path == tmp_path / "guides.jsonl"

    def test_skips_guide_that_fails_all_retries_and_logs(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text("407608\n999\n", encoding="utf-8")
        failing_response = _mock_text_response("")
        failing_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error"
        )
        succeeding_response = _mock_text_response(DOUBLE_QUOTED_GUIDE_PAGE_HTML)

        with patch(
            "requests.get",
            side_effect=[
                failing_response,
                failing_response,
                failing_response,
                succeeding_response,
            ],
        ) as mock_get:
            with patch("src.data_retrieval.play_gwent.downloader.time.sleep"):
                with patch(
                    "src.data_retrieval.play_gwent.downloader.tqdm.write"
                ) as mock_write:
                    guides_path = downloader.phase_2()

        # 3 failed attempts for 407608, then 1 successful attempt for 999.
        assert mock_get.call_count == 4
        mock_write.assert_called_once()
        assert "407608" in mock_write.call_args.args[0]

        manifest_ids = (
            (tmp_path / "guides_manifest.txt").read_text(encoding="utf-8").splitlines()
        )
        assert manifest_ids == ["999"]
        written_lines = guides_path.read_text(encoding="utf-8").splitlines()
        assert len(written_lines) == 1


class TestGetWithRetries:
    def test_returns_immediately_on_first_success(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response("ok")

        with patch("requests.get", return_value=response) as mock_get:
            with patch(
                "src.data_retrieval.play_gwent.downloader.time.sleep"
            ) as mock_sleep:
                result = downloader._get_with_retries("https://example.test/x")

        assert result is response
        mock_get.assert_called_once()
        mock_sleep.assert_not_called()

    def test_retries_then_succeeds(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        failing_response = _mock_text_response("")
        failing_response.raise_for_status.side_effect = requests.ConnectionError(
            "dropped"
        )
        succeeding_response = _mock_text_response("ok")

        with patch(
            "requests.get", side_effect=[failing_response, succeeding_response]
        ) as mock_get:
            with patch(
                "src.data_retrieval.play_gwent.downloader.time.sleep"
            ) as mock_sleep:
                result = downloader._get_with_retries("https://example.test/x")

        assert result is succeeding_response
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(_RETRY_BACKOFF_SECONDS)

    def test_raises_last_error_after_exhausting_all_attempts(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response("")
        response.raise_for_status.side_effect = requests.HTTPError("500")

        with patch("requests.get", return_value=response) as mock_get:
            with patch(
                "src.data_retrieval.play_gwent.downloader.time.sleep"
            ) as mock_sleep:
                with pytest.raises(requests.HTTPError):
                    downloader._get_with_retries("https://example.test/x")

        assert mock_get.call_count == 3
        # No sleep after the final, non-retried attempt.
        assert mock_sleep.call_count == 2
