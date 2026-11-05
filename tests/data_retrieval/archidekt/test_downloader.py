import gzip
import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import requests

from src.data_retrieval.archidekt.downloader import (
    ArchidektDeckDownloader,
    _CrawlPlan,
    _cut_torn_last_line,
    _DeckShards,
    _prune_deck,
    _SpreadOrder,
)
from src.data_retrieval.rate_limiter import RateLimiter

_MODULE = "src.data_retrieval.archidekt.downloader"
_SITE = "https://a.test"


class _FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        return json.loads(self.text)


def _deck_payload(deck_id: int, oracle_uid: str = "o-1") -> dict[str, Any]:
    return {
        "id": deck_id,
        "name": f"Deck {deck_id}",
        "deckFormat": 3,
        "unlisted": deck_id % 2 == 0,
        "owner": {"id": 907, "username": "someone"},
        "featured": "https://img",
        "categories": [{"name": "Commander", "includedInDeck": True}],
        "cards": [
            {
                "id": 11,
                "quantity": 2,
                "modifier": "Normal",
                "categories": ["Commander"],
                "card": {
                    "uid": "p-1",
                    "collectorNumber": "111",
                    "artist": "dropped",
                    "prices": {"ck": 1.0},
                    "edition": {
                        "editioncode": "akh",
                        "editionname": "Amonkhet",
                        "x": 1,
                    },
                    "oracleCard": {
                        "uid": oracle_uid,
                        "name": "Card",
                        "text": "dropped",
                    },
                },
            }
        ],
    }


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


class TestSpreadOrder:
    def test_for_range_visits_every_id_once(self) -> None:
        order = _SpreadOrder.for_range(1000)
        assert sorted(order.id_at(k) for k in range(1000)) == list(range(1, 1001))

    def test_for_range_of_one(self) -> None:
        assert _SpreadOrder.for_range(1).id_at(0) == 1

    def test_for_range_stride_is_coprime_example(self) -> None:
        assert _SpreadOrder.for_range(10) == _SpreadOrder(10, 7, 0)

    @pytest.mark.parametrize(
        "n, stride, offset",
        [(0, 1, 0), (10, 2, 0), (10, 0, 0), (10, 3, 10), (10, 3, -1)],
    )
    def test_invalid_orders_raise(self, n: int, stride: int, offset: int) -> None:
        with pytest.raises(ValueError):
            _SpreadOrder(n, stride, offset)

    def test_id_at_outside_the_range_raises(self) -> None:
        with pytest.raises(IndexError):
            _SpreadOrder(3, 1, 0).id_at(3)


class TestCrawlPlan:
    def test_sitemap_ids_first_then_the_spread_without_them(self) -> None:
        plan = _CrawlPlan((7, 2), _SpreadOrder(3, 1, 0))
        # Spread slot 3 holds id 2, a sitemap id: skipped, position kept
        assert list(plan.ids_from(0)) == [(0, 7), (1, 2), (2, 1), (4, 3)]
        assert plan.length == 5

    def test_ids_from_a_middle_position_keeps_positions(self) -> None:
        plan = _CrawlPlan((7, 2), _SpreadOrder(3, 1, 0))
        assert list(plan.ids_from(3)) == [(4, 3)]

    def test_round_trips_through_json(self) -> None:
        plan = _CrawlPlan((7, 2), _SpreadOrder.for_range(100))
        assert _CrawlPlan.from_json(plan.to_json()) == plan
        assert _CrawlPlan.from_json(plan.to_json()).fingerprint == plan.fingerprint

    @pytest.mark.parametrize("sitemap_ids", [(1, 1), (0,), (-3,)])
    def test_invalid_sitemap_ids_raise(self, sitemap_ids: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            _CrawlPlan(sitemap_ids, _SpreadOrder(3, 1, 0))

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            "{}",
            json.dumps(
                {"sitemap_ids": ["1"], "spread": {"n": 3, "stride": 1, "offset": 0}}
            ),
            json.dumps(
                {"sitemap_ids": [], "spread": {"n": 4, "stride": 2, "offset": 0}}
            ),
        ],
    )
    def test_malformed_plans_raise(self, text: str) -> None:
        with pytest.raises(ValueError):
            _CrawlPlan.from_json(text)


class TestPruneDeck:
    def test_keeps_api_names_and_drops_bulk_and_owner(self) -> None:
        row = _prune_deck(_deck_payload(4))
        assert row == {
            "id": 4,
            "name": "Deck 4",
            "deckFormat": 3,
            "unlisted": True,
            "categories": [{"name": "Commander", "includedInDeck": True}],
            "cards": [
                {
                    "id": 11,
                    "quantity": 2,
                    "modifier": "Normal",
                    "categories": ["Commander"],
                    "card": {
                        "uid": "p-1",
                        "collectorNumber": "111",
                        "edition": {"editioncode": "akh", "editionname": "Amonkhet"},
                        "oracleCard": {"uid": "o-1", "name": "Card"},
                    },
                }
            ],
        }

    def test_docstring_example(self) -> None:
        assert _prune_deck({"id": 1, "cards": [], "owner": {"id": 9}}) == {
            "id": 1,
            "cards": [],
        }

    @pytest.mark.parametrize(
        "payload, error",
        [
            ({"cards": []}, KeyError),
            ({"id": 1}, KeyError),
            ({"id": 1, "cards": {}}, TypeError),
            ({"id": 1, "cards": [{"quantity": 1}]}, KeyError),
            ({"id": 1, "cards": [{"card": None}]}, TypeError),
            ([], TypeError),
        ],
    )
    def test_unreadable_decks_raise(self, payload: Any, error: type) -> None:
        with pytest.raises(error):
            _prune_deck(payload)


class TestCutTornLastLine:
    def test_missing_file_is_zero_lines(self, tmp_path: Path) -> None:
        assert _cut_torn_last_line(tmp_path / "none.jsonl") == 0

    def test_whole_lines_are_counted_and_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_bytes(b"a\nb\n")
        assert _cut_torn_last_line(path) == 2
        assert path.read_bytes() == b"a\nb\n"

    def test_a_torn_last_line_is_cut(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_bytes(b'a\nb\n{"par')
        assert _cut_torn_last_line(path) == 2
        assert path.read_bytes() == b"a\nb\n"

    def test_a_file_with_only_a_torn_line_is_emptied(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_bytes(b'{"par')
        assert _cut_torn_last_line(path) == 0
        assert path.read_bytes() == b""


def _gz_text(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8")


class TestDeckShards:
    def test_a_full_shard_is_gzipped_and_the_next_started(self, tmp_path: Path) -> None:
        shards = _DeckShards(tmp_path, rows_per_shard=2)
        for row in ["1", "2", "3"]:
            shards.append(row)
        assert _gz_text(tmp_path / "decks-00000.jsonl.gz") == "1\n2\n"
        assert not (tmp_path / "decks-00000.jsonl").exists()
        assert (tmp_path / "decks-00001.jsonl").read_bytes().decode() == "3\n"

    def test_reopening_continues_the_newest_shard_after_a_torn_line(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "decks-00000.jsonl.gz").write_bytes(gzip.compress(b"1\n2\n"))
        (tmp_path / "decks-00001.jsonl").write_bytes(b"3\n4")  # torn
        shards = _DeckShards(tmp_path, rows_per_shard=2)
        shards.append("5")
        shards.append("6")
        assert _gz_text(tmp_path / "decks-00001.jsonl.gz") == "3\n5\n"
        assert (tmp_path / "decks-00002.jsonl").read_bytes().decode() == "6\n"

    def test_reopening_after_a_crash_mid_compression_finishes_it(
        self, tmp_path: Path
    ) -> None:
        # Older plain shard never compressed; newest compressed, plain left
        (tmp_path / "decks-00000.jsonl").write_bytes(b"1\n2\n")
        (tmp_path / "decks-00001.jsonl.gz").write_bytes(gzip.compress(b"3\n4\n"))
        (tmp_path / "decks-00001.jsonl").write_bytes(b"3\n4\n")
        shards = _DeckShards(tmp_path, rows_per_shard=2)
        shards.append("5")
        assert _gz_text(tmp_path / "decks-00000.jsonl.gz") == "1\n2\n"
        assert not (tmp_path / "decks-00000.jsonl").exists()
        assert not (tmp_path / "decks-00001.jsonl").exists()
        assert (tmp_path / "decks-00002.jsonl").read_bytes().decode() == "5\n"

    def test_rows_per_shard_below_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _DeckShards(tmp_path, rows_per_shard=0)

    def test_a_full_plain_shard_is_gzipped_on_reopen(self, tmp_path: Path) -> None:
        (tmp_path / "decks-00000.jsonl").write_bytes(b"1\n2\n")
        shards = _DeckShards(tmp_path, rows_per_shard=2)
        shards.append("3")
        assert _gz_text(tmp_path / "decks-00000.jsonl.gz") == "1\n2\n"
        assert (tmp_path / "decks-00001.jsonl").read_bytes().decode() == "3\n"


class TestPhase1:
    def test_writes_the_plan_from_the_sitemap_and_newest_id(
        self, tmp_path: Path
    ) -> None:
        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site({})):
            plan_path = _make_downloader(tmp_path).phase_1()

        plan = _CrawlPlan.from_json(plan_path.read_text(encoding="utf-8"))
        assert plan.sitemap_ids == (5, 2)
        assert plan.spread == _SpreadOrder.for_range(6)

    def test_an_existing_plan_is_kept(self, tmp_path: Path) -> None:
        (tmp_path / "crawl_plan.json").write_text("kept")
        with patch(f"{_MODULE}.paced_get") as get:
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()
        assert (tmp_path / "crawl_plan.json").read_bytes().decode() == "kept"

    def test_a_cursor_without_a_plan_raises(self, tmp_path: Path) -> None:
        (tmp_path / "cursor.json").write_text("{}")
        with (
            patch(f"{_MODULE}.paced_get") as get,
            pytest.raises(RuntimeError, match="cursor"),
        ):
            _make_downloader(tmp_path).phase_1()
        get.assert_not_called()


class TestPhase2:
    def _plan(self, raw_dir: Path) -> None:
        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site({})):
            _make_downloader(raw_dir).phase_1()

    def test_walks_the_whole_plan(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        decks = {deck_id: _deck_payload(deck_id) for deck_id in (1, 2, 5, 6)}
        decks[3] = 403
        calls: list[str] = []

        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks, calls=calls)):
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

        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks, calls=first)):
            _make_downloader(tmp_path, max_decks_per_session=3).phase_2()
        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks, calls=second)):
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
            patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks)),
            pytest.raises(requests.ConnectionError),
        ):
            _make_downloader(tmp_path).phase_2()

        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 1
        assert [row["id"] for row in _rows(tmp_path)] == [5]

    def test_a_non_json_200_is_transient(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        decks: dict[int, Any] = {5: "<html>challenge</html>"}

        with (
            patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks)),
            pytest.raises(ValueError, match="not JSON"),
        ):
            _make_downloader(tmp_path).phase_2()

        cursor = json.loads((tmp_path / "cursor.json").read_text())
        assert cursor["position"] == 0 and cursor["failures"] == 1

    def test_an_unreadable_deck_is_rejected_and_the_walk_moves_on(
        self, tmp_path: Path
    ) -> None:
        self._plan(tmp_path)
        decks: dict[int, Any] = {5: {"name": "no id or cards"}, 2: _deck_payload(2)}

        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks)):
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
        plan = _CrawlPlan.from_json((tmp_path / "crawl_plan.json").read_text())
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
                patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks)),
                pytest.raises(requests.ConnectionError),
            ):
                _make_downloader(tmp_path).phase_2()
            cursor = json.loads((tmp_path / "cursor.json").read_text())
            if session < 2:
                assert cursor == {**cursor, "position": 0, "failures": session + 1}

        assert cursor["position"] == 1 and cursor["failures"] == 0
        reject = json.loads((tmp_path / "rejects.jsonl").read_text())
        assert reject["id"] == 5 and reject["reason"].startswith("transient x 3")

        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site(decks)):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()
        assert [row["id"] for row in _rows(tmp_path)] == [2]

    def test_a_success_clears_the_failure_count(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        with (
            patch(
                f"{_MODULE}.paced_get",
                side_effect=_fake_site({5: requests.ConnectionError()}),
            ),
            pytest.raises(requests.ConnectionError),
        ):
            _make_downloader(tmp_path).phase_2()

        with patch(
            f"{_MODULE}.paced_get", side_effect=_fake_site({5: _deck_payload(5)})
        ):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()

        cursor = json.loads((tmp_path / "cursor.json").read_text())
        assert cursor["position"] == 1 and cursor["failures"] == 0

    def test_a_cursor_without_failures_reads_as_none(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        plan = _CrawlPlan.from_json((tmp_path / "crawl_plan.json").read_text())
        (tmp_path / "cursor.json").write_text(
            json.dumps({"position": 2, "plan": plan.fingerprint})
        )
        calls: list[str] = []
        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site({}, calls=calls)):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()
        assert len(calls) == 1
        assert json.loads((tmp_path / "cursor.json").read_text())["position"] == 3

    def test_a_torn_rejects_line_is_cut_before_appending(self, tmp_path: Path) -> None:
        self._plan(tmp_path)
        (tmp_path / "rejects.jsonl").write_bytes(b'{"position": 0, "id": 9}\n{"pos')
        with patch(f"{_MODULE}.paced_get", side_effect=_fake_site({5: 410})):
            _make_downloader(tmp_path, max_decks_per_session=1).phase_2()
        lines = (tmp_path / "rejects.jsonl").read_bytes().decode().splitlines()
        assert [json.loads(line)["id"] for line in lines] == [9, 5]

    def test_without_a_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _make_downloader(tmp_path).phase_2()


def test_max_decks_per_session_below_one_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make_downloader(tmp_path, max_decks_per_session=0)
