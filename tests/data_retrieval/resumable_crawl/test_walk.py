import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pytest
import requests

from src.data_retrieval.resumable_crawl import (
    CrawlBlockedError,
    FetchOutcome,
    ItemFound,
    ItemNotFound,
    ItemRejected,
    ResumableCrawl,
    SessionCounts,
    WalkPolicy,
)


@dataclass(frozen=True)
class _ListPlan:
    """A CrawlPlan over a list; None marks a skipped position."""

    slots: tuple[str | None, ...]
    fingerprint: str = "plan-1"

    @property
    def length(self) -> int:
        return len(self.slots)

    def ids_from(self, position: int) -> Iterator[tuple[int, str]]:
        for walk_position in range(position, len(self.slots)):
            item_id = self.slots[walk_position]
            if item_id is not None:
                yield walk_position, item_id


def _site(
    answers: dict[str, FetchOutcome | Exception],
) -> Callable[[str], FetchOutcome]:
    """A fetch: answers maps id -> outcome or an exception; others found."""

    def _fetch(item_id: str) -> FetchOutcome:
        answer = answers.get(item_id, ItemFound({"id": item_id}))
        if isinstance(answer, Exception):
            raise answer
        return answer

    return _fetch


def _crawl(
    tmp_path: Path, *slots: str | None, policy: WalkPolicy | None = None
) -> ResumableCrawl[str]:
    return ResumableCrawl[str](
        tmp_path, _ListPlan(slots), "Test decks", policy or WalkPolicy()
    )


def _cursor(tmp_path: Path) -> dict[str, int]:
    return json.loads((tmp_path / "cursor.json").read_text())


def _rows(tmp_path: Path) -> list[str]:
    return [
        json.loads(line)["id"]
        for line in (tmp_path / "decks-00000.jsonl").read_text().splitlines()
    ]


def _rejects(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "rejects.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestWalk:
    def test_one_row_per_variant_and_the_cursor_at_the_end(
        self, tmp_path: Path
    ) -> None:
        fetch = _site({"b": ItemNotFound(), "c": ItemRejected("HTTP 410")})

        counts = _crawl(tmp_path, "a", "b", "c").walk(fetch)

        assert counts == SessionCounts(found=1, not_found=1, rejected=1)
        assert _rows(tmp_path) == ["a"]
        assert _rejects(tmp_path) == [{"position": 2, "id": "c", "reason": "HTTP 410"}]
        assert _cursor(tmp_path)["position"] == 3

    def test_max_items_caps_a_session_and_the_next_resumes(
        self, tmp_path: Path
    ) -> None:
        crawl = _crawl(tmp_path, "a", "b", "c")
        crawl.walk(_site({}), max_items=2)
        assert _cursor(tmp_path)["position"] == 2

        crawl.walk(_site({}), max_items=2)
        assert _rows(tmp_path) == ["a", "b", "c"]
        assert _cursor(tmp_path)["position"] == 3

    def test_a_walk_at_the_end_does_nothing(self, tmp_path: Path) -> None:
        crawl = _crawl(tmp_path, "a")
        crawl.walk(_site({}))
        assert crawl.walk(_site({})) == SessionCounts()

    def test_trailing_skips_move_the_cursor_to_the_end(self, tmp_path: Path) -> None:
        _crawl(tmp_path, "a", None, None).walk(_site({}))
        assert _cursor(tmp_path)["position"] == 3

    def test_a_capped_session_does_not_jump_past_live_ids(self, tmp_path: Path) -> None:
        _crawl(tmp_path, "a", None, "c").walk(_site({}), max_items=1)
        assert _cursor(tmp_path)["position"] == 1

    def test_a_transient_failure_keeps_the_cursor_and_counts_it(
        self, tmp_path: Path
    ) -> None:
        crawl = _crawl(tmp_path, "a", "b")
        with pytest.raises(requests.ConnectionError):
            crawl.walk(_site({"b": requests.ConnectionError("down")}))
        assert _cursor(tmp_path) == {"position": 1, "failures": 1, "plan": "plan-1"}

        crawl.walk(_site({}))
        assert _rows(tmp_path) == ["a", "b"]
        assert _cursor(tmp_path)["failures"] == 0

    def test_an_id_failing_every_allowed_session_is_rejected(
        self, tmp_path: Path
    ) -> None:
        crawl = _crawl(tmp_path, "a", "b", policy=WalkPolicy(max_session_failures=2))
        fetch = _site({"a": ValueError("challenge page")})
        for _ in range(2):
            with pytest.raises(ValueError):
                crawl.walk(fetch)

        assert _cursor(tmp_path)["position"] == 1
        assert _rejects(tmp_path)[0]["reason"].startswith("transient x 2")
        crawl.walk(fetch)
        assert _rows(tmp_path) == ["b"]

    def test_a_transient_reject_counts_in_neither_tally(self, tmp_path: Path) -> None:
        crawl = _crawl(
            tmp_path,
            "a",
            "b",
            policy=WalkPolicy(max_session_failures=1, max_consecutive_rejects=1),
        )
        with pytest.raises(ValueError):
            crawl.walk(_site({"a": ValueError("challenge page")}))
        # Rejected by the failure count, not by a fetch: no CrawlBlockedError
        assert crawl.walk(_site({})) == SessionCounts(found=1, not_found=0, rejected=0)

    def test_consecutive_rejects_stop_the_session(self, tmp_path: Path) -> None:
        refused = ItemRejected("HTTP 403")
        crawl = _crawl(
            tmp_path, "a", "b", "c", "d", policy=WalkPolicy(max_consecutive_rejects=2)
        )
        with pytest.raises(CrawlBlockedError):
            crawl.walk(_site({"b": refused, "c": refused}))
        assert _cursor(tmp_path)["position"] == 3
        assert [row["id"] for row in _rejects(tmp_path)] == ["b", "c"]

    def test_a_found_item_resets_the_reject_run(self, tmp_path: Path) -> None:
        refused = ItemRejected("HTTP 403")
        crawl = _crawl(
            tmp_path, "a", "b", "c", policy=WalkPolicy(max_consecutive_rejects=2)
        )
        counts = crawl.walk(_site({"a": refused, "c": refused}))
        assert counts == SessionCounts(found=1, not_found=0, rejected=2)

    def test_a_cursor_of_another_plan_raises(self, tmp_path: Path) -> None:
        (tmp_path / "cursor.json").write_text('{"position": 0, "plan": "other"}')
        with pytest.raises(ValueError, match="another plan"):
            _crawl(tmp_path, "a").walk(_site({}))

    def test_max_items_below_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_items"):
            _crawl(tmp_path, "a").walk(_site({}), max_items=0)


class TestWalkPolicy:
    @pytest.mark.parametrize(
        "failures, rejects", [(0, None), (-1, None), (3, 0), (3, -2)]
    )
    def test_invalid_policies_raise(self, failures: int, rejects: int | None) -> None:
        with pytest.raises(ValueError):
            WalkPolicy(failures, rejects)
