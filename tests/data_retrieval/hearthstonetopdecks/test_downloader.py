import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.hearthstonetopdecks.downloader import (
    HearthstoneTopDecksDownloader,
    unwrap_deck_page,
)
from src.data_retrieval.rate_limiter import RateLimiter

_SITE_URL = "https://example.test"
_GET = "src.data_retrieval.hearthstonetopdecks.downloader.requests.get"
_RETRY_SLEEP = "src.data_retrieval.download_utils.time.sleep"

_DECK_PAGE = """<html><body><main><article>
<div class="class-header"><h1>Murloc Paladin</h1></div>
<div class="space-y-1 text-center text-sm">
<p><span>Posted by<!-- -->: <strong><a href="/user/x/">Hearthston3r</a></strong></span>
<span>Dust: <span><img src="dust.gif" alt="Dust"/>7,040</span></span></p>
<p><span>Class<!-- -->: <strong><a href="/c/">Paladin</a></strong></span>
<span>Format<!-- -->: <strong><a href="/f/">Standard</a></strong></span></p>
</div>
<input readOnly="" aria-label="Deck code" value="AAECAZ8FAAAAAA=="/>
</article></main></body></html>"""


def _make_downloader(raw_data_dir: Path) -> HearthstoneTopDecksDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return HearthstoneTopDecksDownloader(
        RateLimiter(requests_per_minute=1_000_000_000), raw_data_dir, _SITE_URL
    )


def _response(text: str, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(str(status_code))
    return response


def _listing(slugs: list[str]) -> str:
    links = "".join(f'<a href="/decks/{slug}/">{slug}</a>' for slug in slugs)
    return f'<html>{links}<a href="/decks/page/2/">next</a></html>'


class TestPhase1:
    def test_walks_listing_pages_until_a_404(self, tmp_path: Path) -> None:
        responses = [
            _response(_listing(["a", "b", "a"])),
            _response(_listing(["b", "c"])),  # "b" shifted by a new deck
            _response("", status_code=404),
        ]

        with patch(_GET, side_effect=responses) as get:
            result = _make_downloader(tmp_path).phase_1()

        assert result.read_text(encoding="utf-8").split() == [
            f"{_SITE_URL}/decks/{slug}/" for slug in ("a", "b", "c")
        ]
        assert [call.args[0] for call in get.call_args_list] == [
            f"{_SITE_URL}/decks/page/{page}/" for page in (1, 2, 3)
        ]

    def test_stops_at_a_page_without_deck_links(self, tmp_path: Path) -> None:
        responses = [_response(_listing(["a"])), _response(_listing([]))]

        with patch(_GET, side_effect=responses):
            result = _make_downloader(tmp_path).phase_1()

        assert result.read_text(encoding="utf-8").split() == [f"{_SITE_URL}/decks/a/"]

    def test_raises_when_no_decks_listed(self, tmp_path: Path) -> None:
        with patch(_GET, return_value=_response("", status_code=404)):
            with pytest.raises(ValueError, match="no decks listed"):
                _make_downloader(tmp_path).phase_1()

    def test_a_failing_listing_page_raises(self, tmp_path: Path) -> None:
        with patch(_GET, return_value=_response("", status_code=500)), patch(
            _RETRY_SLEEP
        ):
            with pytest.raises(requests.HTTPError):
                _make_downloader(tmp_path).phase_1()


class TestPhase2:
    def test_appends_unwrapped_rows_and_skips_manifested(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        downloader.deck_urls_path.write_text("u/done/\nu/new/\n", encoding="utf-8")
        downloader.decks_manifest_path.write_text("u/done/\n", encoding="utf-8")

        with patch(_GET, return_value=_response(_DECK_PAGE)) as get:
            result = downloader.phase_2()

        assert [call.args[0] for call in get.call_args_list] == ["u/new/"]
        rows = [json.loads(line) for line in result.read_text().splitlines()]
        assert [row["url"] for row in rows] == ["u/new/"]
        assert rows[0]["deck_code"] == "AAECAZ8FAAAAAA=="
        assert downloader.decks_manifest_path.read_text().split() == [
            "u/done/",
            "u/new/",
        ]

    def test_skips_a_failing_page_and_continues(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        downloader.deck_urls_path.write_text("u/bad/\nu/good/\n", encoding="utf-8")

        def _get(url: str, **_: object) -> MagicMock:
            if url == "u/bad/":
                return _response("", status_code=500)
            return _response(_DECK_PAGE)

        with patch(_GET, side_effect=_get), patch(_RETRY_SLEEP):
            result = downloader.phase_2()

        rows = [json.loads(line) for line in result.read_text().splitlines()]
        assert [row["url"] for row in rows] == ["u/good/"]


class TestUnwrapDeckPage:
    def test_reads_title_code_and_details(self) -> None:
        row = unwrap_deck_page("u/a/", _DECK_PAGE)

        assert row == {
            "url": "u/a/",
            "title": "Murloc Paladin",
            "deck_code": "AAECAZ8FAAAAAA==",
            "details": {
                "Posted by": "Hearthston3r",
                "Dust": "7,040",
                "Class": "Paladin",
                "Format": "Standard",
            },
        }

    def test_a_page_without_a_deck_is_all_none(self) -> None:
        row = unwrap_deck_page("u/a/", "<html><body>Not found</body></html>")

        assert row == {"url": "u/a/", "title": None, "deck_code": None, "details": {}}
