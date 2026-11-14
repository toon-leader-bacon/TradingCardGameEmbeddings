import gzip
import json
import logging
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from src.data_retrieval.mtgtop8.crawl_plan import MtgTop8PlanFile
from src.data_retrieval.mtgtop8.downloader import MtgTop8DeckDownloader
from src.data_retrieval.mtgtop8.formats import MtgTop8Format
from src.data_retrieval.mtgtop8.listing import SEARCH_PAGE_ROWS
from src.data_retrieval.rate_limiter import RateLimiter
from tests.data_retrieval.mtgtop8.fixtures import EXPORT, PageDeck, search_page_html

# paced_get as the shared crawl calls it
_PACED_GET = "src.data_retrieval.resumable_crawl.fetch.paced_get"
_SITE = "https://t8.test"


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


def _fake_site(
    listed_ids: list[int],
    exports: dict[int, Any] | None = None,
    calls: list[str] | None = None,
) -> Callable[..., _FakeResponse]:
    """A paced_get stand-in: the search pages list listed_ids, 25 a page;
    exports maps a deck id -> export text, an int status, or an
    exception; other decks answer EXPORT."""

    def _get(url: str, **_: Any) -> _FakeResponse:
        if calls is not None:
            calls.append(url)
        parts = urlsplit(url)
        query = {name: values[0] for name, values in parse_qs(parts.query).items()}
        if parts.path == "/search":
            start = (int(query["current_page"]) - 1) * SEARCH_PAGE_ROWS
            page_ids = listed_ids[start : start + SEARCH_PAGE_ROWS]
            decks = [
                PageDeck(deck_id, format_code=query["format"]) for deck_id in page_ids
            ]
            return _FakeResponse(200, search_page_html(decks))
        answer = (exports or {}).get(int(query["d"]), EXPORT)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return _FakeResponse(answer, "")
        return _FakeResponse(200, answer)

    return _get


def _make_downloader(
    raw_dir: Path, deck_format: MtgTop8Format = MtgTop8Format.PAUPER, **kwargs: Any
) -> MtgTop8DeckDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return MtgTop8DeckDownloader(
        deck_format,
        RateLimiter(requests_per_minute=1_000_000_000),
        raw_dir,
        _SITE,
        **kwargs,
    )


def _rows(format_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(format_dir.glob("decks-*")):
        data = path.read_bytes()
        text = gzip.decompress(data) if path.suffix == ".gz" else data
        rows.extend(json.loads(line) for line in text.decode("utf-8").splitlines())
    return rows


def _search_calls(calls: list[str]) -> list[int]:
    return [
        int(parse_qs(urlsplit(url).query)["current_page"][0])
        for url in calls
        if "/search?" in url
    ]


class TestConstructor:
    def test_defaults_to_robots_txt_pace(self) -> None:
        assert (
            MtgTop8DeckDownloader(
                MtgTop8Format.VINTAGE
            ).rate_limiter.requests_per_minute
            == 12
        )

    def test_a_faster_rate_is_taken_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            downloader = MtgTop8DeckDownloader(MtgTop8Format.VINTAGE, RateLimiter(60))

        assert downloader.rate_limiter.requests_per_minute == 60
        assert "faster than robots.txt" in caplog.text

    def test_each_format_has_its_own_directory(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path, MtgTop8Format.CEDH)

        assert downloader.format_dir == tmp_path / "cEDH"

    def test_max_decks_per_session_below_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_decks_per_session"):
            _make_downloader(tmp_path, max_decks_per_session=0)


class TestPhase1:
    def test_lands_pages_until_a_short_one_then_writes_the_plan(
        self, tmp_path: Path
    ) -> None:
        calls: list[str] = []
        listed = list(range(1, 2 * SEARCH_PAGE_ROWS + 4))  # 2 full pages + 3
        with patch(_PACED_GET, _fake_site(listed, calls=calls)):
            plan_path = _make_downloader(tmp_path).phase_1()

        assert plan_path == tmp_path / "PAU" / "crawl_plan.jsonl"
        assert _search_calls(calls) == [1, 2, 3]
        assert f"{_SITE}/search?format=PAU&current_page=1" in calls
        plan = MtgTop8PlanFile(plan_path)
        assert sorted(deck_id for _, deck_id in plan.ids_from(0)) == listed
        assert sorted(
            path.name for path in (tmp_path / "PAU" / "listing").iterdir()
        ) == [
            "page-00001.jsonl",
            "page-00002.jsonl",
            "page-00003.jsonl",
        ]

    def test_an_exact_multiple_ends_on_an_empty_page(self, tmp_path: Path) -> None:
        calls: list[str] = []
        with patch(_PACED_GET, _fake_site(list(range(1, 26)), calls=calls)):
            _make_downloader(tmp_path).phase_1()

        assert _search_calls(calls) == [1, 2]

    def test_resumes_after_the_last_landed_page(self, tmp_path: Path) -> None:
        listed = list(range(1, 2 * SEARCH_PAGE_ROWS + 4))
        failing = _fake_site(listed)

        def _fail_on_page_2(url: str, **kwargs: Any) -> _FakeResponse:
            if "current_page=2" in url:
                raise requests.ConnectionError("down")
            return failing(url, **kwargs)

        with patch(_PACED_GET, _fail_on_page_2), pytest.raises(
            requests.ConnectionError
        ):
            _make_downloader(tmp_path).phase_1()
        calls: list[str] = []
        with patch(_PACED_GET, _fake_site(listed, calls=calls)):
            _make_downloader(tmp_path).phase_1()

        assert _search_calls(calls) == [2, 3]

    def test_a_complete_listing_is_not_fetched_again(self, tmp_path: Path) -> None:
        with patch(_PACED_GET, _fake_site([1, 2])):
            plan_path = _make_downloader(tmp_path).phase_1()
        plan_path.unlink()  # e.g. a crash before the plan was written
        calls: list[str] = []
        with patch(_PACED_GET, _fake_site([1, 2], calls=calls)):
            _make_downloader(tmp_path).phase_1()

        assert calls == []
        assert plan_path.is_file()

    def test_an_existing_plan_is_kept(self, tmp_path: Path) -> None:
        with patch(_PACED_GET, _fake_site([1, 2])):
            plan_path = _make_downloader(tmp_path).phase_1()
        before = plan_path.read_bytes()
        calls: list[str] = []
        with patch(_PACED_GET, _fake_site([1, 2, 3], calls=calls)):
            _make_downloader(tmp_path).phase_1()

        assert calls == []
        assert plan_path.read_bytes() == before


class TestPhase2:
    def test_lands_each_export_with_its_listing_row(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        with patch(_PACED_GET, _fake_site([7, 8, 9], exports={8: "", 9: 403})):
            downloader.phase_1()
            format_dir = downloader.phase_2()

        rows = _rows(format_dir)
        assert rows == [
            {
                "deck_id": 7,
                "event_id": 93623,
                "format": "PAU",
                "deck_name": "Elves",
                "event_name": "Bournemouth League @ The Trove (Bournemouth, UK)",
                "rank": "2",
                "date": "09/11/26",
                "mtgo": EXPORT,
            }
        ]
        rejects = (format_dir / "rejects.jsonl").read_text(encoding="utf-8")
        assert '"HTTP 403"' in rejects and "9" in rejects

    def test_a_body_that_is_not_an_export_is_transient(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        with patch(_PACED_GET, _fake_site([7], exports={7: "<html>busy</html>"})):
            downloader.phase_1()
            with pytest.raises(ValueError, match="unreadable"):
                downloader.phase_2()

        assert _rows(tmp_path / "PAU") == []

    def test_max_decks_per_session_caps_a_session(self, tmp_path: Path) -> None:
        with patch(_PACED_GET, _fake_site([7, 8, 9])):
            downloader = _make_downloader(tmp_path, max_decks_per_session=2)
            downloader.phase_1()
            downloader.phase_2()
            assert len(_rows(tmp_path / "PAU")) == 2
            downloader.phase_2()

        assert sorted(row["deck_id"] for row in _rows(tmp_path / "PAU")) == [7, 8, 9]
