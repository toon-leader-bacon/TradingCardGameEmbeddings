import pytest

from src.data_retrieval.deckstats.crawl_plan import (
    DECKSTATS_ITEMS,
    deck_id_of_path,
    keep_first_path_per_id,
    parse_deck_sitemap_urls,
    parse_sitemap_deck_paths,
)
from src.data_retrieval.resumable_crawl import SitemapThenSpreadPlan, SpreadOrder


class TestDeckIdOfPath:
    @pytest.mark.parametrize(
        "path, deck_id",
        [
            ("325", 325),
            ("325-Burn-Mana-Burn", 325),
            ("/decks/1705661", 1705661),
            ("/decks/1705660-Middle-School", 1705660),
        ],
    )
    def test_reads_every_form(self, path: str, deck_id: int) -> None:
        assert deck_id_of_path(path) == deck_id

    @pytest.mark.parametrize("path", ["", "Burn", "-325", "/decks/", "325/x", "/x/325"])
    def test_other_paths_raise(self, path: str) -> None:
        with pytest.raises(ValueError):
            deck_id_of_path(path)


class TestParseDeckSitemapUrls:
    def test_deck_sitemaps_only_in_order_without_repeats(self) -> None:
        site = "https://deckstats.net/sitemaps"
        index = (
            f"<loc>{site}/sitemap-core-00.xml</loc>"
            f"<loc>{site}/sitemap-decks-qualified-00.xml</loc>"
            f"<loc>{site}/sitemap-lorcana-decks-00.xml</loc>"
            f"<loc>{site}/sitemap-decks-fresh-00.xml</loc>"
            f"<loc>{site}/sitemap-decks-qualified-00.xml</loc>"
        )
        assert parse_deck_sitemap_urls(index) == [
            f"{site}/sitemap-decks-qualified-00.xml",
            f"{site}/sitemap-decks-fresh-00.xml",
        ]

    def test_no_deck_sitemaps_raise(self) -> None:
        with pytest.raises(ValueError):
            parse_deck_sitemap_urls("<loc>https://deckstats.net/sitemaps/x.xml</loc>")


class TestParseSitemapDeckPaths:
    def test_paths_in_order(self) -> None:
        xml = (
            "<url><loc>https://deckstats.net/decks/325-Burn-Mana-Burn</loc>"
            "<lastmod>2012-12-06</lastmod></url>"
            "<url><loc> https://deckstats.net/decks/350-Sharuum </loc></url>"
            "<url><loc>https://deckstats.net/help</loc></url>"
            "<url><loc>https://deckstats.net/decks/9-x?print=1</loc></url>"
        )
        assert parse_sitemap_deck_paths(xml) == ["325-Burn-Mana-Burn", "350-Sharuum"]

    def test_an_empty_sitemap_has_no_paths(self) -> None:
        assert parse_sitemap_deck_paths("<urlset></urlset>") == []


class TestKeepFirstPathPerId:
    def test_docstring_example(self) -> None:
        assert keep_first_path_per_id(["2-Burn", "3-Elves", "2-Burn-v2"]) == (
            "2-Burn",
            "3-Elves",
        )

    def test_empty(self) -> None:
        assert keep_first_path_per_id([]) == ()


class TestDeckstatsPlan:
    def test_sitemap_paths_then_bare_ids_without_sitemap_ids(self) -> None:
        plan = SitemapThenSpreadPlan(("2-Burn",), SpreadOrder(3, 1, 0), DECKSTATS_ITEMS)
        assert list(plan.ids_from(0)) == [(0, "2-Burn"), (1, "1"), (3, "3")]

    def test_over_sitemap_spreads_to_the_highest_id(self) -> None:
        plan = SitemapThenSpreadPlan.over_sitemap(
            ("7-Burn", "3-Elves"), DECKSTATS_ITEMS
        )
        assert plan.spread.n == 7 and plan.length == 9

    def test_round_trips_through_json(self) -> None:
        plan = SitemapThenSpreadPlan.over_sitemap(("7-Burn",), DECKSTATS_ITEMS)
        assert plan.to_json() == (
            '{"sitemap_paths": ["7-Burn"], "spread": {"n": 7, "stride": 4, "offset": 0}}'
        )
        parsed = SitemapThenSpreadPlan.from_json(plan.to_json(), DECKSTATS_ITEMS)
        assert parsed == plan and parsed.fingerprint == plan.fingerprint

    @pytest.mark.parametrize("paths", [("2-Burn", "2-Burn-v2"), ("Burn",), ("0-Zero",)])
    def test_invalid_sitemap_paths_raise(self, paths: tuple[str, ...]) -> None:
        with pytest.raises(ValueError):
            SitemapThenSpreadPlan(paths, SpreadOrder(3, 1, 0), DECKSTATS_ITEMS)

    def test_a_plan_with_int_paths_is_malformed(self) -> None:
        text = '{"sitemap_paths": [7], "spread": {"n": 3, "stride": 1, "offset": 0}}'
        with pytest.raises(ValueError):
            SitemapThenSpreadPlan.from_json(text, DECKSTATS_ITEMS)
