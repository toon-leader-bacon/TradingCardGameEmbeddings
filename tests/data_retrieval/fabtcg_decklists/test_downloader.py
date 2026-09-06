import xml.etree.ElementTree as ElementTree
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.fabtcg_decklists.downloader import (
    _BROWSER_USER_AGENT,
    _DECK_URL_PATTERN,
    _DECKLIST_SITEMAP_PATTERN,
    FabtcgDecklistDownloader,
)
from src.data_retrieval.rate_limiter import RateLimiter

_SITEMAP_INDEX_URL = "https://example.test/sitemap_index.xml"

_DECK_URL_A = "https://fabtcg.com/decklists/tom-penny-go-wide-warrior-deck/"
_DECK_URL_B = "https://fabtcg.com/decklists/john-jaurigue-warrior-deck/"

_FRAGMENT_HTML = (
    '<section class="decklist-list-view block hidden">'
    '<div class="card-name"><span>1x</span> Dorinthea Ironsong</div>'
    "</section>"
)


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(output_dir: Path) -> FabtcgDecklistDownloader:
    return FabtcgDecklistDownloader(
        _fast_rate_limiter(),
        output_dir,
        _SITEMAP_INDEX_URL,
    )


def _sitemap_xml(locs: list[str]) -> str:
    loc_elements = "".join(f"<url><loc>{loc}</loc></url>" for loc in locs)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{loc_elements}"
        "</urlset>"
    )


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = text
    return response


def _page_html(fragment_html: str = _FRAGMENT_HTML) -> str:
    return f"<html><body>{fragment_html}</body></html>"


class TestInit:
    def test_defaults_urls_and_output_dir_when_omitted(self) -> None:
        downloader = FabtcgDecklistDownloader(_fast_rate_limiter())

        assert downloader.output_dir == FabtcgDecklistDownloader.DEFAULT_RAW_DATA_DIR
        assert (
            downloader.sitemap_index_url
            == FabtcgDecklistDownloader.DEFAULT_SITEMAP_INDEX_URL
        )

    def test_honors_explicit_overrides(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        assert downloader.output_dir == tmp_path
        assert downloader.sitemap_index_url == _SITEMAP_INDEX_URL


class TestExtractMatchingLocUrls:
    def test_extracts_urls_matching_pattern_in_document_order(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        sitemap_xml = _sitemap_xml(
            [
                "https://fabtcg.com/post-sitemap.xml",
                "https://fabtcg.com/decklist-sitemap.xml",
                "https://fabtcg.com/decklist-sitemap2.xml",
            ]
        )

        urls = downloader._extract_matching_loc_urls(
            sitemap_xml, _DECKLIST_SITEMAP_PATTERN
        )

        assert urls == [
            "https://fabtcg.com/decklist-sitemap.xml",
            "https://fabtcg.com/decklist-sitemap2.xml",
        ]

    def test_excludes_bare_and_locale_prefixed_decklists_entries(
        self, tmp_path: Path
    ) -> None:
        # Regression guard: the real decklist sitemaps also list a bare
        # "/decklists/" entry (no slug) and locale-prefixed variants —
        # neither is a real deck page, both must be excluded.
        downloader = _make_downloader(tmp_path)
        sitemap_xml = _sitemap_xml(
            [
                "https://fabtcg.com/decklists/",
                "https://fabtcg.com/fr/decklists/tom-penny-go-wide-warrior-deck/",
                "https://fabtcg.com/ja/decklists/tom-penny-go-wide-warrior-deck/",
                _DECK_URL_A,
            ]
        )

        urls = downloader._extract_matching_loc_urls(sitemap_xml, _DECK_URL_PATTERN)

        assert urls == [_DECK_URL_A]

    def test_returns_empty_list_for_sitemap_with_no_locs(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        sitemap_xml = _sitemap_xml([])

        urls = downloader._extract_matching_loc_urls(sitemap_xml, _DECK_URL_PATTERN)

        assert urls == []

    def test_raises_parse_error_on_invalid_xml(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(ElementTree.ParseError):
            downloader._extract_matching_loc_urls("<not-valid-xml", _DECK_URL_PATTERN)


class TestExtractDeckSlug:
    def test_extracts_slug_from_deck_url(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        slug = downloader._extract_deck_slug(_DECK_URL_A)

        assert slug == "tom-penny-go-wide-warrior-deck"

    def test_raises_on_url_not_matching_expected_shape(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(ValueError, match="does not match"):
            downloader._extract_deck_slug(
                "https://fabtcg.com/fr/decklists/tom-penny-go-wide-warrior-deck/"
            )


class TestExtractDecklistFragment:
    def test_extracts_and_prettifies_the_fragment(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        fragment = downloader._extract_decklist_fragment(_page_html())

        assert "Dorinthea Ironsong" in fragment
        assert fragment.startswith('<section class="decklist-list-view block hidden">')

    def test_extracts_from_real_fixture_fragment(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        fixture_path = Path(__file__).parents[3] / "data" / "tmp" / "list_view.html"
        if not fixture_path.exists():
            pytest.skip("real fixture not present in this environment")

        fragment = downloader._extract_decklist_fragment(fixture_path.read_text())

        assert "Dorinthea Ironsong" in fragment
        assert "Pitch 1" in fragment
        assert "Pitch 2" in fragment
        assert "Pitch 3" in fragment

    def test_raises_when_fragment_not_found(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(ValueError, match="No element matching"):
            downloader._extract_decklist_fragment(
                "<html><body>no deck here</body></html>"
            )


class TestPhase1:
    def test_writes_deck_urls_from_index_and_sub_sitemaps(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        index_response = _mock_text_response(
            _sitemap_xml(
                [
                    "https://fabtcg.com/post-sitemap.xml",
                    "https://fabtcg.com/decklist-sitemap.xml",
                ]
            )
        )
        sub_sitemap_response = _mock_text_response(
            _sitemap_xml([_DECK_URL_A, _DECK_URL_B])
        )

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if url == _SITEMAP_INDEX_URL:
                return index_response
            return sub_sitemap_response

        with patch("requests.get", side_effect=_get_side_effect) as mock_get:
            urls_path = downloader.phase_1()

        assert mock_get.call_count == 2
        assert urls_path == tmp_path / "deck_urls.txt"
        assert (
            urls_path.read_text(encoding="utf-8") == f"{_DECK_URL_A}\n{_DECK_URL_B}\n"
        )

    def test_sends_browser_user_agent_header(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        index_response = _mock_text_response(
            _sitemap_xml(["https://fabtcg.com/decklist-sitemap.xml"])
        )
        sub_sitemap_response = _mock_text_response(_sitemap_xml([_DECK_URL_A]))

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if url == _SITEMAP_INDEX_URL:
                return index_response
            return sub_sitemap_response

        with patch("requests.get", side_effect=_get_side_effect) as mock_get:
            downloader.phase_1()

        assert mock_get.call_count == 2
        for call in mock_get.call_args_list:
            assert call.kwargs["headers"] == {"User-Agent": _BROWSER_USER_AGENT}

    def test_deduplicates_urls_preserving_first_seen_order(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)
        index_response = _mock_text_response(
            _sitemap_xml(["https://fabtcg.com/decklist-sitemap.xml"])
        )
        sub_sitemap_response = _mock_text_response(
            _sitemap_xml([_DECK_URL_A, _DECK_URL_B, _DECK_URL_A])
        )

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if url == _SITEMAP_INDEX_URL:
                return index_response
            return sub_sitemap_response

        with patch("requests.get", side_effect=_get_side_effect):
            urls_path = downloader.phase_1()

        assert (
            urls_path.read_text(encoding="utf-8") == f"{_DECK_URL_A}\n{_DECK_URL_B}\n"
        )

    def test_raises_when_no_deck_urls_found(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_text_response(_sitemap_xml([]))

        with patch("requests.get", return_value=response):
            with pytest.raises(ValueError, match="No deck URLs"):
                downloader.phase_1()

    def test_raises_when_request_fails_after_retries(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch("requests.get", side_effect=requests.ConnectionError("down")):
            with pytest.raises(requests.ConnectionError):
                downloader.phase_1()

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        index_response = _mock_text_response(
            _sitemap_xml(["https://fabtcg.com/decklist-sitemap.xml"])
        )
        sub_sitemap_response = _mock_text_response(_sitemap_xml([_DECK_URL_A]))

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if url == _SITEMAP_INDEX_URL:
                return index_response
            return sub_sitemap_response

        with patch("requests.get", side_effect=_get_side_effect):
            downloader.phase_1()

        assert nested_dir.is_dir()


class TestPhase2:
    def test_raises_if_phase_1_has_not_run(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with pytest.raises(FileNotFoundError):
            downloader.phase_2()

    def test_saves_each_deck_not_already_on_disk(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_urls.txt").write_text(
            f"{_DECK_URL_A}\n{_DECK_URL_B}\n", encoding="utf-8"
        )
        response = _mock_text_response(_page_html())

        with patch("requests.get", return_value=response) as mock_get:
            decklists_dir = downloader.phase_2()

        assert mock_get.call_count == 2
        assert decklists_dir == tmp_path / "decklists"
        assert (decklists_dir / "tom-penny-go-wide-warrior-deck.html").exists()
        assert (decklists_dir / "john-jaurigue-warrior-deck.html").exists()
        assert "Dorinthea Ironsong" in (
            decklists_dir / "tom-penny-go-wide-warrior-deck.html"
        ).read_text(encoding="utf-8")

    def test_sends_browser_user_agent_header(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_urls.txt").write_text(f"{_DECK_URL_A}\n", encoding="utf-8")
        response = _mock_text_response(_page_html())

        with patch("requests.get", return_value=response) as mock_get:
            downloader.phase_2()

        assert mock_get.call_args.kwargs["headers"] == {
            "User-Agent": _BROWSER_USER_AGENT
        }

    def test_skips_a_deck_already_saved_to_disk(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_urls.txt").write_text(
            f"{_DECK_URL_A}\n{_DECK_URL_B}\n", encoding="utf-8"
        )
        decklists_dir = tmp_path / "decklists"
        decklists_dir.mkdir()
        (decklists_dir / "tom-penny-go-wide-warrior-deck.html").write_text(
            "already here", encoding="utf-8"
        )
        response = _mock_text_response(_page_html())

        with patch("requests.get", return_value=response) as mock_get:
            downloader.phase_2()

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == _DECK_URL_B
        assert (decklists_dir / "tom-penny-go-wide-warrior-deck.html").read_text(
            encoding="utf-8"
        ) == "already here"

    def test_skips_and_logs_a_deck_that_fails_after_retries(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_urls.txt").write_text(
            f"{_DECK_URL_A}\n{_DECK_URL_B}\n", encoding="utf-8"
        )
        good_response = _mock_text_response(_page_html())

        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            if url == _DECK_URL_A:
                raise requests.ConnectionError("down")
            return good_response

        with patch("requests.get", side_effect=_get_side_effect):
            decklists_dir = downloader.phase_2()

        assert not (decklists_dir / "tom-penny-go-wide-warrior-deck.html").exists()
        assert (decklists_dir / "john-jaurigue-warrior-deck.html").exists()
        assert _DECK_URL_A in capsys.readouterr().out

    def test_skips_and_logs_a_deck_with_no_decklist_fragment(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_urls.txt").write_text(f"{_DECK_URL_A}\n", encoding="utf-8")
        response = _mock_text_response("<html><body>no deck here</body></html>")

        with patch("requests.get", return_value=response):
            decklists_dir = downloader.phase_2()

        assert not (decklists_dir / "tom-penny-go-wide-warrior-deck.html").exists()
        assert _DECK_URL_A in capsys.readouterr().out

    def test_all_urls_already_saved_makes_no_requests(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        (tmp_path / "deck_urls.txt").write_text(f"{_DECK_URL_A}\n", encoding="utf-8")
        decklists_dir = tmp_path / "decklists"
        decklists_dir.mkdir()
        (decklists_dir / "tom-penny-go-wide-warrior-deck.html").write_text(
            "already here", encoding="utf-8"
        )

        with patch("requests.get") as mock_get:
            result = downloader.phase_2()

        mock_get.assert_not_called()
        assert result == decklists_dir
