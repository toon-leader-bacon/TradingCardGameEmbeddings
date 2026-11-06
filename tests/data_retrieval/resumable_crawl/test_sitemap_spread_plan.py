import pytest

from src.data_retrieval.archidekt.crawl_plan import ARCHIDEKT_ITEMS
from src.data_retrieval.resumable_crawl import SitemapThenSpreadPlan, SpreadOrder


class TestSpreadOrderDict:
    def test_round_trips(self) -> None:
        order = SpreadOrder.for_range(100)
        assert SpreadOrder.from_dict(order.to_dict()) == order
        assert list(order.to_dict()) == ["n", "stride", "offset"]

    @pytest.mark.parametrize(
        "raw, error",
        [
            ({"n": 3, "stride": 1}, KeyError),
            ({"n": "3", "stride": 1, "offset": 0}, TypeError),
            ([3, 1, 0], TypeError),
            ({"n": 4, "stride": 2, "offset": 0}, ValueError),
        ],
    )
    def test_malformed_raises(self, raw: object, error: type) -> None:
        with pytest.raises(error):
            SpreadOrder.from_dict(raw)


class TestOverSitemap:
    def test_spreads_over_the_highest_sitemap_id(self) -> None:
        plan = SitemapThenSpreadPlan.over_sitemap((3, 9), ARCHIDEKT_ITEMS)
        assert plan.spread == SpreadOrder.for_range(9) and plan.length == 11

    def test_no_sitemap_items_raise(self) -> None:
        with pytest.raises(ValueError):
            SitemapThenSpreadPlan.over_sitemap((), ARCHIDEKT_ITEMS)


def test_a_plan_with_only_sitemap_ids_skips_every_spread_slot() -> None:
    plan = SitemapThenSpreadPlan((2, 1), SpreadOrder(2, 1, 0), ARCHIDEKT_ITEMS)
    assert list(plan.ids_from(0)) == [(0, 2), (1, 1)]
    assert list(plan.ids_from(2)) == []
