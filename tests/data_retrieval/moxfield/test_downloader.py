import gzip
import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import requests

from src.data_retrieval.moxfield.crawl_plan import MoxfieldPlanFile
from src.data_retrieval.moxfield.downloader import MoxfieldDeckDownloader
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.resumable_crawl import CrawlBlockedError
from tests.data_retrieval.moxfield.fixtures import deck_payload

# paced_get as the shared crawl calls it
_PACED_GET = "src.data_retrieval.resumable_crawl.fetch.paced_get"
_SITE = "https://m.test"
_API = "https://api.m.test"
_A = "KtNrenwexES3cLg2NDMEnw"
_B = "4Q2TsGQTbUa71hN41hlbww"
_C = "Qapw_ySpn0Co2ZV7k4QlNg"

_INDEX = f"""<sitemapindex>
  <sitemap><loc>{_SITE}/sitemap-static.xml</loc></sitemap>
  <sitemap><loc>{_SITE}/sitemap-deck-1.xml</loc></sitemap>
  <sitemap><loc>{_SITE}/sitemap-deck-2.xml</loc></sitemap>
</sitemapindex>"""
_DECK_SITEMAPS = {
    1: f"<url><loc>{_SITE}/decks/{_A}</loc><lastmod>2022-12-22</lastmod></url>"
    f"<url><loc>{_SITE}/decks/{_A}/primer</loc></url>",
    2: f"<url><loc>{_SITE}/decks/{_B}</loc><lastmod>2026-05-04</lastmod></url>"
    f"<url><loc>{_SITE}/decks/{_C}</loc></url>",
}


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


def _fake_site(
    decks: dict[str, Any] | None = None, calls: list[str] | None = None
) -> Callable[..., _FakeResponse]:
    """A paced_get stand-in. decks maps public id -> payload, an int
    status, or an exception; other ids answer their fixture payload."""

    def _get(url: str, **_: Any) -> _FakeResponse:
        if calls is not None:
            calls.append(url)
        if url == f"{_SITE}/sitemap.xml":
            return _FakeResponse(200, _INDEX)
        if url.startswith(f"{_SITE}/sitemap-deck-"):
            number = int(url.rsplit("-", 1)[-1].removesuffix(".xml"))
            return _FakeResponse(200, _DECK_SITEMAPS[number])
        public_id = url.rsplit("/", 1)[-1]
        answer = (decks or {}).get(public_id, deck_payload(public_id))
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return _FakeResponse(answer, '{"error": "x"}')
        if isinstance(answer, str):
            return _FakeResponse(200, answer)
        return _FakeResponse(200, json.dumps(answer))

    return _get


def _make_downloader(raw_dir: Path, **kwargs: Any) -> MoxfieldDeckDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return MoxfieldDeckDownloader(
        RateLimiter(requests_per_minute=1_000_000_000), raw_dir, _SITE, _API, **kwargs
    )


def _rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("decks-*")):
        data = path.read_bytes()
        text = gzip.decompress(data) if path.suffix == ".gz" else data
        rows.extend(json.loads(line) for line in text.decode("utf-8").splitlines())
    return rows


def _planned(raw_dir: Path) -> list[str]:
    plan = MoxfieldPlanFile(raw_dir / "crawl_plan.tsv")
    return [public_id for _, public_id in plan.ids_from(0)]


class TestPhase1:
    def test_lands_every_deck_sitemap_and_writes_a_shuffled_plan(
        self, tmp_path: Path
    ) -> None:
        with patch(_PACED_GET, side_effect=_fake_site()):
            plan_path = _make_downloader(tmp_path).phase_1()

        assert plan_path == tmp_path / "crawl_plan.tsv"
        assert sorted(_planned(tmp_path)) == sorted([_A, _B, _C])
        assert (tmp_path / "sitemaps" / "deck-00001.tsv").read_text() == (
            f"{_A}\t2022-12-22\n"
        )
        assert (tmp_path / "sitemaps" / "deck-00002.tsv").read_text() == (
            f"{_B}\t2026-05-04\n{_C}\t\n"
        )

    def test_landed_sitemaps_are_not_fetched_again(self, tmp_path: Path) -> None:
        sitemaps = tmp_path / "sitemaps"
        sitemaps.mkdir()
        (sitemaps / "deck-00001.tsv").write_text(f"{_A}\t2022-12-22\n")
        calls: list[str] = []
        with patch(_PACED_GET, side_effect=_fake_site(calls=calls)):
            _make_downloader(tmp_path).phase_1()
        assert calls == [f"{_SITE}/sitemap.xml", f"{_SITE}/sitemap-deck-2.xml"]

    def test_a_deck_sitemap_answering_404_is_skipped(self, tmp_path: Path) -> None:
        site = _fake_site()

        def _get(url: str, **kwargs: Any) -> _FakeResponse:
            if url.endswith("sitemap-deck-2.xml"):
                return _FakeResponse(404, "gone")
            return site(url, **kwargs)

        with patch(_PACED_GET, side_effect=_get):
            _make_downloader(tmp_path).phase_1()
        assert _planned(tmp_path) == [_A]
        assert not (tmp_path / "sitemaps" / "deck-00002.tsv").exists()

    def test_an_existing_plan_is_kept(self, tmp_path: Path) -> None:
        (tmp_path / "crawl_plan.tsv").write_text(f"{_A}\t\n")
        with patch(_PACED_GET) as get:
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()

    def test_a_cursor_without_a_plan_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "cursor.json").write_text("{}")
        with patch(_PACED_GET) as get, pytest.raises(RuntimeError):
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()


class TestPhase2:
    def _plan(self, tmp_path: Path) -> list[str]:
        with patch(_PACED_GET, side_effect=_fake_site()):
            _make_downloader(tmp_path).phase_1()
        return _planned(tmp_path)

    def test_lands_one_pruned_row_per_deck_in_plan_order(self, tmp_path: Path) -> None:
        planned = self._plan(tmp_path)
        calls: list[str] = []
        with patch(_PACED_GET, side_effect=_fake_site(calls=calls)):
            assert _make_downloader(tmp_path).phase_2() == tmp_path

        assert calls == [f"{_API}/v3/decks/all/{public_id}" for public_id in planned]
        rows = _rows(tmp_path)
        assert [row["publicId"] for row in rows] == planned
        assert "createdByUser" not in rows[0] and "ownerUserId" not in rows[0]
        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 3

    def test_a_session_stops_at_max_decks_and_the_next_resumes(
        self, tmp_path: Path
    ) -> None:
        planned = self._plan(tmp_path)
        with patch(_PACED_GET, side_effect=_fake_site()):
            _make_downloader(tmp_path, max_decks_per_session=2).phase_2()
            assert len(_rows(tmp_path)) == 2
            _make_downloader(tmp_path, max_decks_per_session=2).phase_2()
        assert [row["publicId"] for row in _rows(tmp_path)] == planned

    def test_404s_land_nothing_and_403s_are_rejected(self, tmp_path: Path) -> None:
        first, second, third = self._plan(tmp_path)
        with patch(_PACED_GET, side_effect=_fake_site({first: 404, second: 403})):
            _make_downloader(tmp_path).phase_2()
        assert [row["publicId"] for row in _rows(tmp_path)] == [third]
        rejects = (tmp_path / "rejects.jsonl").read_text().splitlines()
        assert [json.loads(line) for line in rejects] == [
            {"position": 1, "id": second, "reason": "HTTP 403"}
        ]

    def test_a_challenge_page_is_transient(self, tmp_path: Path) -> None:
        first, *_ = self._plan(tmp_path)
        with (
            patch(_PACED_GET, side_effect=_fake_site({first: "<html>wait</html>"})),
            pytest.raises(ValueError, match="unreadable"),
        ):
            _make_downloader(tmp_path).phase_2()
        cursor = json.loads((tmp_path / "cursor.json").read_text())
        assert cursor["position"] == 0 and cursor["failures"] == 1

    def test_a_network_error_is_transient(self, tmp_path: Path) -> None:
        first, *_ = self._plan(tmp_path)
        with (
            patch(
                _PACED_GET,
                side_effect=_fake_site({first: requests.ConnectionError("down")}),
            ),
            pytest.raises(requests.ConnectionError),
        ):
            _make_downloader(tmp_path).phase_2()
        assert json.loads((tmp_path / "cursor.json").read_text())["failures"] == 1

    def test_twenty_rejects_in_a_row_stop_the_crawl(self, tmp_path: Path) -> None:
        public_ids = [f"{index:022d}" for index in range(25)]
        (tmp_path / "crawl_plan.tsv").write_text(
            "".join(f"{public_id}\t\n" for public_id in public_ids)
        )
        refused = {public_id: 403 for public_id in public_ids}
        with (
            patch(_PACED_GET, side_effect=_fake_site(refused)),
            pytest.raises(CrawlBlockedError),
        ):
            _make_downloader(tmp_path).phase_2()
        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 20

    def test_without_a_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _make_downloader(tmp_path).phase_2()


def test_max_decks_per_session_below_one_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make_downloader(tmp_path, max_decks_per_session=0)
