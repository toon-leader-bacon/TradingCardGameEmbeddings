import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import requests

from src.data_retrieval.archidekt.crawl_plan import ARCHIDEKT_ITEMS
from src.data_retrieval.resumable_crawl import SitemapThenSpreadPlan, SpreadOrder
from src.data_retrieval.archidekt.downloader import ArchidektDeckDownloader
from src.data_retrieval.rate_limiter import RateLimiter
from tests.data_retrieval.archidekt.fixtures import deck_payload as _deck_payload

# paced_get as the shared crawl calls it
_PACED_GET = "src.data_retrieval.resumable_crawl.fetch.paced_get"
_SITE = "https://a.test"


class _FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        return json.loads(self.text)


_SITEMAP = """<?xml version="1.0"?><urlset>
  <url>
    <loc>https://archidekt.com/decks/5</loc>
  </url>
  <url><loc>https://archidekt.com/decks/2</loc></url>
  <url><loc>https://archidekt.com/decks/5</loc></url>
</urlset>"""


def _make_downloader(raw_dir: Path, **kwargs: Any) -> ArchidektDeckDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    return ArchidektDeckDownloader(
        RateLimiter(requests_per_minute=1_000_000_000), raw_dir, _SITE, **kwargs
    )


def _fake_site(
    decks: dict[int, Any], newest_id: int = 6, calls: list[str] | None = None
) -> Callable[..., _FakeResponse]:
    """A paced_get stand-in: decks maps id -> payload, an int status, or
    an exception to raise; unknown ids are 404."""

    def _get(url: str, **_: Any) -> _FakeResponse:
        if calls is not None:
            calls.append(url)
        if url.endswith("/sitemaps/decks.xml"):
            return _FakeResponse(200, _SITEMAP)
        if "/api/decks/v3/" in url:
            return _FakeResponse(200, {"results": [{"id": newest_id}, {"id": 3}]})
        deck_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        answer = decks.get(deck_id, 404)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return _FakeResponse(answer, {"error": "x"})
        return _FakeResponse(200, answer)

    return _get


def _rows(raw_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in sorted(raw_dir.glob("decks-*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class TestPhase1:
    def test_writes_the_plan_from_the_sitemap_and_newest_id(
        self, tmp_path: Path
    ) -> None:
        with patch(_PACED_GET, side_effect=_fake_site({})):
            plan_path = _make_downloader(tmp_path).phase_1()

        plan = SitemapThenSpreadPlan.from_json(
            plan_path.read_text(encoding="utf-8"), ARCHIDEKT_ITEMS
        )
        assert plan.sitemap_items == (5, 2)
        assert plan.spread == SpreadOrder.for_range(6)

    def test_an_existing_plan_is_kept(self, tmp_path: Path) -> None:
        (tmp_path / "crawl_plan.json").write_text("kept")
        with patch(_PACED_GET) as get:
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()
        assert (tmp_path / "crawl_plan.json").read_bytes().decode() == "kept"

    def test_a_cursor_without_a_plan_raises(self, tmp_path: Path) -> None:
        (tmp_path / "cursor.json").write_text("{}")
        with (
            patch(_PACED_GET) as get,
            pytest.raises(RuntimeError, match="cursor"),
        ):
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()


class TestPhase2:
    def _plan(self, raw_dir: Path) -> None:
        with patch(_PACED_GET, side_effect=_fake_site({})):
            _make_downloader(raw_dir).phase_1()

    def test_walks_the_whole_plan(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        decks = {deck_id: _deck_payload(deck_id) for deck_id in (1, 2, 5, 6)}
        decks[3] = 403
        calls: list[str] = []

        with patch(_PACED_GET, side_effect=_fake_site(decks, calls=calls)):
            _make_downloader(tmp_path).phase_2()

        deck_calls = [int(url.rstrip("/").rsplit("/", 1)[-1]) for url in calls]
        assert deck_calls[:2] == [5, 2]
        assert sorted(deck_calls) == [1, 2, 3, 4, 5, 6]
        assert sorted(row["id"] for row in _rows(tmp_path)) == [1, 2, 5, 6]
        rejects = (tmp_path / "rejects.jsonl").read_text().splitlines()
        assert [json.loads(line)["id"] for line in rejects] == [3]
        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 8

    def test_sessions_resume_where_the_last_stopped(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        decks = {deck_id: _deck_payload(deck_id) for deck_id in range(1, 7)}
        first: list[str] = []
        second: list[str] = []

        with patch(_PACED_GET, side_effect=_fake_site(decks, calls=first)):
            _make_downloader(tmp_path, max_decks_per_session=3).phase_2()
        with patch(_PACED_GET, side_effect=_fake_site(decks, calls=second)):
            _make_downloader(tmp_path).phase_2()

        assert len(first) == 3
        assert set(first).isdisjoint(second)
        assert sorted(row["id"] for row in _rows(tmp_path)) == [1, 2, 3, 4, 5, 6]

    def test_a_transient_failure_raises_and_keeps_the_cursor(
        self, tmp_path: Path
    ) -> None:
        self._plan(tmp_path)
        decks: dict[int, Any] = {5: _deck_payload(5), 2: requests.ConnectionError()}

        with (
            patch(_PACED_GET, side_effect=_fake_site(decks)),
            pytest.raises(requests.ConnectionError),
        ):
            _make_downloader(tmp_path).phase_2()

        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 1
        assert [row["id"] for row in _rows(tmp_path)] == [5]

    def test_a_non_json_200_is_transient(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        decks: dict[int, Any] = {5: "<html>challenge</html>"}

        with (
            patch(_PACED_GET, side_effect=_fake_site(decks)),
            pytest.raises(ValueError, match="unreadable"),
        ):
            _make_downloader(tmp_path).phase_2()

        cursor = json.loads((tmp_path / "cursor.json").read_text())
        assert cursor["position"] == 0 and cursor["failures"] == 1

    def test_an_unreadable_deck_is_rejected_and_the_walk_moves_on(
        self, tmp_path: Path
    ) -> None:
        self._plan(tmp_path)
        decks: dict[int, Any] = {5: {"name": "no id or cards"}, 2: _deck_payload(2)}

        with patch(_PACED_GET, side_effect=_fake_site(decks)):
            _make_downloader(tmp_path, max_decks_per_session=2).phase_2()

        reject = json.loads((tmp_path / "rejects.jsonl").read_text())
        assert reject["id"] == 5 and "unreadable" in reject["reason"]
        assert [row["id"] for row in _rows(tmp_path)] == [2]

    def test_a_cursor_from_another_plan_raises(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        (tmp_path / "cursor.json").write_text(json.dumps({"position": 0, "plan": "x"}))
        with pytest.raises(ValueError, match="another"):
            _make_downloader(tmp_path).phase_2()

    def test_a_cursor_past_the_plan_raises(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        plan = SitemapThenSpreadPlan.from_json(
            (tmp_path / "crawl_plan.json").read_text(), ARCHIDEKT_ITEMS
        )
        (tmp_path / "cursor.json").write_text(
            json.dumps({"position": plan.length + 1, "plan": plan.fingerprint})
        )
        with pytest.raises(ValueError, match="position"):
            _make_downloader(tmp_path).phase_2()

    def test_an_id_failing_three_sessions_is_rejected_and_skipped(
        self, tmp_path: Path
    ) -> None:
        self._plan(tmp_path)
        decks: dict[int, Any] = {
            5: requests.ConnectionError("down"),
            2: _deck_payload(2),
        }

        for session in range(3):
            with (
                patch(_PACED_GET, side_effect=_fake_site(decks)),
                pytest.raises(requests.ConnectionError),
            ):
                _make_downloader(tmp_path).phase_2()
            cursor = json.loads((tmp_path / "cursor.json").read_text())
            if session < 2:
                assert cursor == {**cursor, "position": 0, "failures": session + 1}

        assert cursor["position"] == 1 and cursor["failures"] == 0
        reject = json.loads((tmp_path / "rejects.jsonl").read_text())
        assert reject["id"] == 5 and reject["reason"].startswith("transient x 3")

        with patch(_PACED_GET, side_effect=_fake_site(decks)):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()
        assert [row["id"] for row in _rows(tmp_path)] == [2]

    def test_a_success_clears_the_failure_count(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        with (
            patch(
                _PACED_GET,
                side_effect=_fake_site({5: requests.ConnectionError()}),
            ),
            pytest.raises(requests.ConnectionError),
        ):
            _make_downloader(tmp_path).phase_2()

        with patch(_PACED_GET, side_effect=_fake_site({5: _deck_payload(5)})):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()

        cursor = json.loads((tmp_path / "cursor.json").read_text())
        assert cursor["position"] == 1 and cursor["failures"] == 0

    def test_a_cursor_without_failures_reads_as_none(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        plan = SitemapThenSpreadPlan.from_json(
            (tmp_path / "crawl_plan.json").read_text(), ARCHIDEKT_ITEMS
        )
        (tmp_path / "cursor.json").write_text(
            json.dumps({"position": 2, "plan": plan.fingerprint})
        )
        calls: list[str] = []
        with patch(_PACED_GET, side_effect=_fake_site({}, calls=calls)):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()
        assert len(calls) == 1
        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 3

    def test_a_torn_rejects_line_is_cut_before_appending(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        (tmp_path / "rejects.jsonl").write_bytes(b'{"position": 0, "id": 9}\n{"pos')
        with patch(_PACED_GET, side_effect=_fake_site({5: 410})):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()
        lines = (tmp_path / "rejects.jsonl").read_bytes().decode().splitlines()
        assert [json.loads(line)["id"] for line in lines] == [9, 5]

    def test_without_a_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _make_downloader(tmp_path).phase_2()


def test_max_decks_per_session_below_one_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make_downloader(tmp_path, max_decks_per_session=0)
