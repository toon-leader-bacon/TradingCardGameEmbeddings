import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.pitchstack.downloader import PitchstackDeckDownloader
from src.data_retrieval.rate_limiter import RateLimiter

_SITEMAP_URL = "https://example.test/sitemaps/public-decks-0.xml"
_DECK_URL = "https://example.test/decks/{deck_id}"

_DECK_ID_A = "d-f947a7e9-699b-5938-9db0-2edca16c6c40"
_DECK_ID_B = "d-1a2b3c4d-1234-5678-9abc-def012345678"


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(raw_data_dir: Path) -> PitchstackDeckDownloader:
    return PitchstackDeckDownloader(
        _fast_rate_limiter(),
        raw_data_dir,
        _SITEMAP_URL,
        _DECK_URL,
    )


def _sitemap_xml(deck_urls: list[str]) -> str:
    locs = "".join(f"<url><loc>{url}</loc></url>" for url in deck_urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{locs}"
        "</urlset>"
    )


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = text
    return response


class TestInit:
    def test_defaults_urls_and_output_dir_when_omitted(self) -> None:
        downloader = PitchstackDeckDownloader(_fast_rate_limiter())

        assert downloader.raw_data_dir == PitchstackDeckDownloader.DEFAULT_RAW_DATA_DIR
        assert downloader.sitemap_url == PitchstackDeckDownloader.DEFAULT_SITEMAP_URL
        assert downloader.deck_url == PitchstackDeckDownloader.DEFAULT_DECK_URL

    def test_honors_explicit_overrides(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        assert downloader.raw_data_dir == tmp_path
        assert downloader.sitemap_url == _SITEMAP_URL
        assert downloader.deck_url == _DECK_URL


class TestExtractDeckIds:
    def test_extracts_ids_from_loc_urls_in_document_order(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        sitemap_xml = _sitemap_xml(
            [
                f"https://pitchstack.gg/decks/{_DECK_ID_A}",
                f"https://pitchstack.gg/decks/{_DECK_ID_B}",
            ]
        )

        ids = downloader._extract_deck_ids(sitemap_xml)

        assert ids == [_DECK_ID_A, _DECK_ID_B]

    def test_skips_loc_urls_with_no_deck_id(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        sitemap_xml = _sitemap_xml(
            ["https://pitchstack.gg/about", f"https://pitchstack.gg/decks/{_DECK_ID_A}"]
        )

        ids = downloader._extract_deck_ids(sitemap_xml)

        assert ids == [_DECK_ID_A]

    def test_does_not_match_a_deck_version_id(self, tmp_path: Path) -> None:
        # Regression guard: a deck *version* id ("dv-<uuid>") must not
        # be mistaken for a deck id ("d-<uuid>") — the two are distinct
        # ids this source deliberately does not conflate (see module
        # docstring).
        downloader = _make_downloader(tmp_path)
        sitemap_xml = _sitemap_xml(
            ["https://pitchstack.gg/decks/dv-5ada809a-9288-5de5-bbb1-9f78a5265bd4"]
        )

        ids = downloader._extract_deck_ids(sitemap_xml)

        assert ids == []

    def test_returns_empty_list_for_sitemap_with_no_locs(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        sitemap_xml = _sitemap_xml([])

        ids = downloader._extract_deck_ids(sitemap_xml)

        assert ids == []

    def test_raises_parse_error_on_invalid_xml(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(ElementTree.ParseError):
            downloader._extract_deck_ids("<not-valid-xml")


class TestPhase1:
    def test_writes_ids_from_sitemap(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response(
            _sitemap_xml(
                [
                    f"https://pitchstack.gg/decks/{_DECK_ID_A}",
                    f"https://pitchstack.gg/decks/{_DECK_ID_B}",
                ]
            )
        )

        with patch("requests.get", return_value=response) as mock_get:
            ids_path = downloader.phase_1()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == _SITEMAP_URL
        assert ids_path == tmp_path / "deck_ids.txt"
        assert ids_path.read_text(encoding="utf-8") == f"{_DECK_ID_A}\n{_DECK_ID_B}\n"

    def test_deduplicates_ids_preserving_first_seen_order(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response(
            _sitemap_xml(
                [
                    f"https://pitchstack.gg/decks/{_DECK_ID_A}",
                    f"https://pitchstack.gg/decks/{_DECK_ID_B}",
                    f"https://pitchstack.gg/decks/{_DECK_ID_A}",
                ]
            )
        )

        with patch("requests.get", return_value=response):
            ids_path = downloader.phase_1()

        assert ids_path.read_text(encoding="utf-8") == f"{_DECK_ID_A}\n{_DECK_ID_B}\n"

    def test_raises_when_sitemap_has_no_deck_ids(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response(_sitemap_xml(["https://pitchstack.gg/about"]))

        with patch("requests.get", return_value=response):
            with pytest.raises(ValueError, match="No deck ids"):
                downloader.phase_1()

    def test_raises_when_request_fails_after_retries(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch("requests.get", side_effect=requests.ConnectionError("down")):
            with pytest.raises(requests.ConnectionError):
                downloader.phase_1()

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        response = _mock_text_response(
            _sitemap_xml([f"https://pitchstack.gg/decks/{_DECK_ID_A}"])
        )

        with patch("requests.get", return_value=response):
            downloader.phase_1()

        assert nested_dir.is_dir()


class TestPhase2:
    def test_raises_if_phase_1_has_not_run(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(FileNotFoundError):
            downloader.phase_2()

    def test_appends_each_deck_not_already_in_manifest(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text(
            f"{_DECK_ID_A}\n{_DECK_ID_B}\n", encoding="utf-8"
        )

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            deck_id = _DECK_ID_A if _DECK_ID_A in url else _DECK_ID_B
            return _mock_text_response(json.dumps({"deck": {"id": deck_id}}))

        with patch("requests.get", side_effect=_get_side_effect) as mock_get:
            decks_path = downloader.phase_2()

        assert mock_get.call_count == 2
        assert decks_path == tmp_path / "decks.jsonl"

        rows = [
            json.loads(line)
            for line in decks_path.read_text(encoding="utf-8").splitlines()
        ]
        assert rows[0]["deck"]["id"] == _DECK_ID_A
        assert rows[1]["deck"]["id"] == _DECK_ID_B

        manifest_ids = (
            (tmp_path / "decks_manifest.txt").read_text(encoding="utf-8").splitlines()
        )
        assert manifest_ids == [_DECK_ID_A, _DECK_ID_B]

    def test_skips_a_deck_already_in_manifest(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text(
            f"{_DECK_ID_A}\n{_DECK_ID_B}\n", encoding="utf-8"
        )
        (tmp_path / "decks_manifest.txt").write_text(
            f"{_DECK_ID_A}\n", encoding="utf-8"
        )
        response = _mock_text_response(json.dumps({"deck": {"id": _DECK_ID_B}}))

        with patch("requests.get", return_value=response) as mock_get:
            downloader.phase_2()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == f"https://example.test/decks/{_DECK_ID_B}"

    def test_skips_and_logs_a_deck_that_fails_after_retries(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text(
            f"{_DECK_ID_A}\n{_DECK_ID_B}\n", encoding="utf-8"
        )
        good_response = _mock_text_response(json.dumps({"deck": {"id": _DECK_ID_B}}))

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if _DECK_ID_A in url:
                raise requests.ConnectionError("down")
            return good_response

        with patch("requests.get", side_effect=_get_side_effect):
            decks_path = downloader.phase_2()

        rows = [
            json.loads(line)
            for line in decks_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 1
        manifest_ids = (
            (tmp_path / "decks_manifest.txt").read_text(encoding="utf-8").splitlines()
        )
        assert manifest_ids == [_DECK_ID_B]
        assert _DECK_ID_A in capsys.readouterr().out

    def test_all_ids_already_downloaded_makes_no_requests(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_ids.txt").write_text(f"{_DECK_ID_A}\n", encoding="utf-8")
        (tmp_path / "decks_manifest.txt").write_text(
            f"{_DECK_ID_A}\n", encoding="utf-8"
        )

        with patch("requests.get") as mock_get:
            decks_path = downloader.phase_2()

        mock_get.assert_not_called()
        assert decks_path == tmp_path / "decks.jsonl"
