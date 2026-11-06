import json
from pathlib import Path

import pytest

from src.data_retrieval.archidekt.crawl_plan import ARCHIDEKT_ITEMS
from src.data_retrieval.resumable_crawl import SitemapThenSpreadPlan, SpreadOrder
from src.data_retrieval.resumable_crawl import CursorFile, RejectLog


class TestSpreadOrder:
    def test_for_range_visits_every_id_once(self) -> None:
        order = SpreadOrder.for_range(1000)
        assert sorted(order.id_at(k) for k in range(1000)) == list(range(1, 1001))

    def test_for_range_of_one(self) -> None:
        assert SpreadOrder.for_range(1).id_at(0) == 1

    def test_for_range_stride_is_coprime_example(self) -> None:
        assert SpreadOrder.for_range(10) == SpreadOrder(10, 7, 0)

    @pytest.mark.parametrize(
        "n, stride, offset",
        [(0, 1, 0), (10, 2, 0), (10, 0, 0), (10, 3, 10), (10, 3, -1)],
    )
    def test_invalid_orders_raise(self, n: int, stride: int, offset: int) -> None:
        with pytest.raises(ValueError):
            SpreadOrder(n, stride, offset)

    def test_id_at_outside_the_range_raises(self) -> None:
        with pytest.raises(IndexError):
            SpreadOrder(3, 1, 0).id_at(3)


class TestCrawlPlan:
    def test_sitemap_ids_first_then_the_spread_without_them(self) -> None:
        plan = SitemapThenSpreadPlan((7, 2), SpreadOrder(3, 1, 0), ARCHIDEKT_ITEMS)
        # Spread slot 3 holds id 2, a sitemap id: skipped, position kept
        assert list(plan.ids_from(0)) == [(0, 7), (1, 2), (2, 1), (4, 3)]
        assert plan.length == 5

    def test_ids_from_a_middle_position_keeps_positions(self) -> None:
        plan = SitemapThenSpreadPlan((7, 2), SpreadOrder(3, 1, 0), ARCHIDEKT_ITEMS)
        assert list(plan.ids_from(3)) == [(4, 3)]

    def test_round_trips_through_json(self) -> None:
        plan = SitemapThenSpreadPlan(
            (7, 2), SpreadOrder.for_range(100), ARCHIDEKT_ITEMS
        )
        assert SitemapThenSpreadPlan.from_json(plan.to_json(), ARCHIDEKT_ITEMS) == plan
        assert (
            SitemapThenSpreadPlan.from_json(plan.to_json(), ARCHIDEKT_ITEMS).fingerprint
            == plan.fingerprint
        )

    @pytest.mark.parametrize("sitemap_ids", [(1, 1), (0,), (-3,)])
    def test_invalid_sitemap_ids_raise(self, sitemap_ids: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            SitemapThenSpreadPlan(sitemap_ids, SpreadOrder(3, 1, 0), ARCHIDEKT_ITEMS)

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
            SitemapThenSpreadPlan.from_json(text, ARCHIDEKT_ITEMS)


class TestOnDiskFormat:
    """Pinned bytes: the running crawl's crawl_plan.json and cursor.json
    were written by this format, so a change here strands it."""

    _PLAN = SitemapThenSpreadPlan((7, 2), SpreadOrder(3, 1, 0), ARCHIDEKT_ITEMS)
    _FINGERPRINT = "e6b2cebaecca5412a0903d4bc3238d7ea5db6cad0a740205d6c57bfc16d2a720"

    def test_plan_json_and_fingerprint(self) -> None:
        assert self._PLAN.to_json() == (
            '{"sitemap_ids": [7, 2], "spread": {"n": 3, "stride": 1, "offset": 0}}'
        )
        assert self._PLAN.fingerprint == self._FINGERPRINT

    def test_cursor_and_reject_bytes(self, tmp_path: Path) -> None:
        cursor = tmp_path / "cursor.json"
        CursorFile(cursor, self._PLAN.fingerprint, self._PLAN.length).write(2, 1)
        assert cursor.read_bytes() == (
            b'{"position": 2, "failures": 1, "plan": "'
            + self._FINGERPRINT.encode()
            + b'"}'
        )
        rejects = tmp_path / "rejects.jsonl"
        RejectLog(rejects).append(3, 42, "HTTP 410")
        assert (
            rejects.read_bytes() == b'{"position": 3, "id": 42, "reason": "HTTP 410"}\n'
        )
