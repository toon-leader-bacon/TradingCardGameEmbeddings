import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests

from src.data_retrieval.deckstats.crawl_plan import DECKSTATS_ITEMS
from src.data_retrieval.deckstats.downloader import DeckstatsDeckDownloader
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.resumable_crawl import SitemapThenSpreadPlan, SpreadOrder
from tests.data_retrieval.deckstats.fixtures import deck_page, page_html, private_page

# paced_get as the shared crawl calls it
_PACED_GET = "src.data_retrieval.resumable_crawl.fetch.paced_get"
_SITE = "https://d.test"

_INDEX = f"""<sitemapindex>
  <sitemap><loc>{_SITE}/sitemaps/sitemap-core-00.xml</loc></sitemap>
  <sitemap><loc>{_SITE}/sitemaps/sitemap-decks-qualified-00.xml</loc></sitemap>
  <sitemap><loc>{_SITE}/sitemaps/sitemap-lorcana-decks-00.xml</loc></sitemap>
  <sitemap><loc>{_SITE}/sitemaps/sitemap-decks-fresh-00.xml</loc></sitemap>
</sitemapindex>"""
_SITEMAPS = {
    "sitemap-decks-qualified-00.xml": f"<url><loc>{_SITE}/decks/4-Burn</loc></url>"
    f"<url><loc>{_SITE}/decks/2-Elves</loc></url>",
    "sitemap-decks-fresh-00.xml": f"<url><loc>{_SITE}/decks/4-Burn-Renamed</loc></url>",
}


@dataclass
class _FakeResponse:
    status_code: int
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class _FakeSite:
    """A paced_get stand-in. pages maps a /decks/ path tail to a response;
    a bare id or a stale slug of a known deck redirects to its page."""

    pages: dict[str, _FakeResponse | Exception] = field(default_factory=dict)
    calls: list[tuple[str, bool]] = field(default_factory=list)

    def __call__(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((url, kwargs.get("follow_redirects", True)))
        if url == f"{_SITE}/sitemap.xml":
            return _FakeResponse(200, _INDEX)
        if "/sitemaps/" in url:
            return _FakeResponse(200, _SITEMAPS[url.rsplit("/", 1)[-1]])
        tail = url.split("/decks/", 1)[1]
        if tail in self.pages:
            answer = self.pages[tail]
            if isinstance(answer, Exception):
                raise answer
            return answer
        return _FakeResponse(404)


def _deck(deck_id: int, name: str = "Burn") -> _FakeResponse:
    return _FakeResponse(200, page_html(deck_page(deck_id, name)))


def _redirect(location: str, status: int = 301) -> _FakeResponse:
    return _FakeResponse(status, "", {"Location": location})


def _make_downloader(raw_dir: Path, **kwargs: Any) -> DeckstatsDeckDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return DeckstatsDeckDownloader(
        RateLimiter(requests_per_minute=1_000_000_000), raw_dir, _SITE, **kwargs
    )


def _rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("decks-*")):
        data = path.read_bytes()
        text = gzip.decompress(data) if path.suffix == ".gz" else data
        rows.extend(json.loads(line) for line in text.decode("utf-8").splitlines())
    return rows


def _rejects(raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / "rejects.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_plan(raw_dir: Path, paths: tuple[str, ...], n: int) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan = SitemapThenSpreadPlan(paths, SpreadOrder(n, 1, 0), DECKSTATS_ITEMS)
    (raw_dir / "crawl_plan.json").write_text(plan.to_json())


class TestPhase1:
    def test_plans_deck_sitemaps_first_listing_per_id_then_ids(
        self, tmp_path: Path
    ) -> None:
        site = _FakeSite()
        with patch(_PACED_GET, side_effect=site):
            plan_path = _make_downloader(tmp_path).phase_1()

        plan = SitemapThenSpreadPlan.from_json(plan_path.read_text(), DECKSTATS_ITEMS)
        assert plan.sitemap_items == ("4-Burn", "2-Elves")
        assert plan.spread.n == 4
        fetched = [url for url, _ in site.calls]
        assert not any("lorcana" in url or "core" in url for url in fetched)

    def test_an_existing_plan_is_kept(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, ("4-Burn",), 4)
        with patch(_PACED_GET) as get:
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()

    def test_a_cursor_without_a_plan_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "cursor.json").write_text("{}")
        with patch(_PACED_GET) as get, pytest.raises(RuntimeError):
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()


class TestPhase2:
    def test_a_current_sitemap_path_is_one_unfollowed_get(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, ("4-Burn",), 1)
        site = _FakeSite({"4-Burn": _deck(4), "1": _FakeResponse(404)})
        with patch(_PACED_GET, side_effect=site):
            _make_downloader(tmp_path).phase_2()

        assert site.calls == [
            (f"{_SITE}/decks/4-Burn", False),
            (f"{_SITE}/decks/1", False),
        ]
        rows = _rows(tmp_path)
        assert [row["deck"]["idsdeck"] for row in rows] == [4]
        assert "owner_name" not in rows[0]["deck"]

    def test_a_bare_id_follows_one_paced_redirect(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, (), 1)
        site = _FakeSite({"1": _redirect("/decks/1-Burn"), "1-Burn": _deck(1)})
        with patch(_PACED_GET, side_effect=site):
            _make_downloader(tmp_path).phase_2()

        assert site.calls == [
            (f"{_SITE}/decks/1", False),
            (f"{_SITE}/decks/1-Burn", False),
        ]
        assert [row["deck"]["idsdeck"] for row in _rows(tmp_path)] == [1]

    def test_a_renamed_sitemap_deck_follows_its_absolute_redirect(
        self, tmp_path: Path
    ) -> None:
        _write_plan(tmp_path, ("1-Old",), 1)
        site = _FakeSite(
            {"1-Old": _redirect(f"{_SITE}/decks/1-New"), "1-New": _deck(1, "New")}
        )
        with patch(_PACED_GET, side_effect=site):
            _make_downloader(tmp_path).phase_2()
        assert [row["deck"]["name"] for row in _rows(tmp_path)] == ["New"]

    def test_a_private_deck_lands_in_private(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, (), 1)
        page = _FakeResponse(200, page_html(private_page(1, "Einkaufsliste")))
        with patch(_PACED_GET, side_effect=_FakeSite({"1": page})):
            _make_downloader(tmp_path).phase_2()

        assert _rows(tmp_path) == []
        assert _rows(tmp_path / "private") == [
            {"idsdeck": 1, "deckName": "Einkaufsliste", "visibility": "private"}
        ]

    @pytest.mark.parametrize(
        "pages, reason",
        [
            ({"1": _redirect("https://elsewhere.test/decks/1-x")}, "HTTP 301 to"),
            ({"1": _redirect("/login")}, "HTTP 301 to '/login'"),
            ({"1": _FakeResponse(302)}, "HTTP 302 to ''"),
            ({"1": _FakeResponse(403)}, "HTTP 403"),
            (
                {
                    "1": _FakeResponse(
                        200, page_html({"component": "Home", "props": {}})
                    )
                },
                "page Home",
            ),
        ],
    )
    def test_answers_it_cannot_land_are_rejected(
        self, tmp_path: Path, pages: dict[str, _FakeResponse | Exception], reason: str
    ) -> None:
        _write_plan(tmp_path, (), 1)
        with patch(_PACED_GET, side_effect=_FakeSite(pages)):
            _make_downloader(tmp_path).phase_2()
        assert _rejects(tmp_path)[0]["reason"].startswith(reason)
        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 1

    def test_a_second_redirect_is_rejected_after_two_gets(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, (), 1)
        site = _FakeSite({"1": _redirect("/decks/1-a"), "1-a": _redirect("/decks/1-b")})
        with patch(_PACED_GET, side_effect=site):
            _make_downloader(tmp_path).phase_2()
        assert len(site.calls) == 2
        assert _rejects(tmp_path)[0]["reason"] == "HTTP 301 to '/decks/1-b'"

    def test_a_challenge_page_is_transient(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, (), 1)
        challenge = _FakeResponse(200, "<html>Just a moment...</html>")
        with (
            patch(_PACED_GET, side_effect=_FakeSite({"1": challenge})),
            pytest.raises(ValueError, match="unreadable"),
        ):
            _make_downloader(tmp_path).phase_2()
        cursor = json.loads((tmp_path / "cursor.json").read_text())
        assert cursor["position"] == 0 and cursor["failures"] == 1

    def test_a_network_error_is_transient(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, (), 1)
        down = requests.ConnectionError("down")
        with (
            patch(_PACED_GET, side_effect=_FakeSite({"1": down})),
            pytest.raises(requests.ConnectionError),
        ):
            _make_downloader(tmp_path).phase_2()

    def test_a_session_stops_at_max_decks(self, tmp_path: Path) -> None:
        _write_plan(tmp_path, ("4-Burn", "2-Elves"), 1)
        site = _FakeSite({"4-Burn": _deck(4), "2-Elves": _deck(2)})
        with patch(_PACED_GET, side_effect=site):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()
        assert len(site.calls) == 1
        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 1

    def test_without_a_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _make_downloader(tmp_path).phase_2()


class TestDeckRedirectTarget:
    @pytest.mark.parametrize(
        "location, expected",
        [
            ("/decks/325-Burn", f"{_SITE}/decks/325-Burn"),
            (f"{_SITE}/decks/325-Burn", f"{_SITE}/decks/325-Burn"),
            ("325-Burn", f"{_SITE}/decks/325-Burn"),
            ("decks/325-Burn", f"{_SITE}/decks/decks/325-Burn"),
            ("https://elsewhere.test/decks/325", None),
            ("//elsewhere.test/decks/325", None),
            ("http://d.test/decks/325", None),
            ("/decks/325-Burn?export_txt=1", None),
            ("/decks/325-Burn#cards", None),
            (f"{_SITE}/decks/../login", None),
            ("/login", None),
            ("", None),
        ],
    )
    def test_only_this_sites_plain_deck_pages(
        self, location: str, expected: str | None
    ) -> None:
        downloader = _make_downloader(Path("unused"))
        assert downloader._deck_redirect_target(f"{_SITE}/decks/325", location) == (
            expected
        )


def test_max_decks_per_session_below_one_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make_downloader(tmp_path, max_decks_per_session=0)
