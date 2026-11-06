from pathlib import Path

import pytest

from src.data_retrieval.moxfield.crawl_plan import (
    MoxfieldPlanFile,
    SitemapDeck,
    parse_deck_sitemap_urls,
    merge_and_shuffle,
    parse_sitemap_decks,
    write_sitemap_decks,
)

_A = "KtNrenwexES3cLg2NDMEnw"
_B = "4Q2TsGQTbUa71hN41hlbww"
_C = "Qapw_ySpn0Co2ZV7k4QlNg"

_SITEMAP = f"""<urlset>
  <url><loc>https://moxfield.com/decks/{_A}</loc><lastmod>2022-12-22</lastmod></url>
  <url><loc>https://moxfield.com/decks/{_A}/primer</loc><lastmod>2022-12-22</lastmod></url>
  <url>
    <loc>https://moxfield.com/decks/{_B}</loc>
  </url>
  <url><loc>https://moxfield.com/decks/{_A}</loc><lastmod>2026-01-01</lastmod></url>
</urlset>"""


class TestParseSitemapDecks:
    def test_decks_in_order_without_primers_or_repeats(self) -> None:
        assert parse_sitemap_decks(_SITEMAP) == [
            SitemapDeck(_A, "2022-12-22"),
            SitemapDeck(_B, ""),
        ]

    def test_an_empty_sitemap_has_no_decks(self) -> None:
        assert parse_sitemap_decks("<urlset></urlset>") == []


class TestParseDeckSitemapUrls:
    def test_deck_sitemaps_by_number_skipping_others(self) -> None:
        index = (
            "<loc>https://moxfield.com/sitemap-static.xml</loc>"
            "<loc>https://moxfield.com/sitemap-deck-10.xml</loc>"
            "<loc>https://moxfield.com/sitemap-user-1.xml</loc>"
            "<loc>https://moxfield.com/sitemap-deck-2.xml</loc>"
        )
        assert parse_deck_sitemap_urls(index) == {
            2: "https://moxfield.com/sitemap-deck-2.xml",
            10: "https://moxfield.com/sitemap-deck-10.xml",
        }

    def test_no_deck_sitemaps_raise(self) -> None:
        with pytest.raises(ValueError):
            parse_deck_sitemap_urls(
                "<loc>https://moxfield.com/sitemap-static.xml</loc>"
            )


class TestSitemapDeckLines:
    def test_round_trips(self) -> None:
        deck = SitemapDeck(_A, "2022-12-22")
        assert SitemapDeck.from_line(deck.to_line() + "\n") == deck

    @pytest.mark.parametrize("line", ["", _A, f"{_A}\tx\ty", "short\t2022-01-01"])
    def test_malformed_lines_raise(self, line: str) -> None:
        with pytest.raises(ValueError):
            SitemapDeck.from_line(line)


class TestMergeAndShuffle:
    def _land(self, tmp_path: Path, name: str, decks: list[SitemapDeck]) -> Path:
        path = tmp_path / name
        write_sitemap_decks(path, decks)
        return path

    def test_each_deck_once_with_its_latest_lastmod(self, tmp_path: Path) -> None:
        first = self._land(tmp_path, "1.tsv", [SitemapDeck(_A, "2022-12-22")])
        second = self._land(
            tmp_path, "2.tsv", [SitemapDeck(_A, "2026-01-01"), SitemapDeck(_B, "")]
        )
        decks = merge_and_shuffle([first, second])
        assert sorted(decks, key=lambda deck: deck.public_id) == [
            SitemapDeck(_B, ""),
            SitemapDeck(_A, "2026-01-01"),
        ]

    def test_the_order_depends_only_on_the_set_of_decks(self, tmp_path: Path) -> None:
        decks = [SitemapDeck(_A, ""), SitemapDeck(_B, ""), SitemapDeck(_C, "")]
        forward = self._land(tmp_path, "f.tsv", decks)
        backward = self._land(tmp_path, "b.tsv", decks[::-1])
        assert merge_and_shuffle([forward]) == merge_and_shuffle([backward])

    def test_no_decks_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            merge_and_shuffle([self._land(tmp_path, "e.tsv", [])])


class TestMoxfieldPlanFile:
    def test_reads_lazily_from_a_position(self, tmp_path: Path) -> None:
        path = tmp_path / "crawl_plan.tsv"
        write_sitemap_decks(
            path,
            [SitemapDeck(_A, "2022-12-22"), SitemapDeck(_B, ""), SitemapDeck(_C, "")],
        )
        plan = MoxfieldPlanFile(path)
        assert plan.length == 3
        assert list(plan.ids_from(0)) == [(0, _A), (1, _B), (2, _C)]
        assert list(plan.ids_from(2)) == [(2, _C)]
        assert list(plan.ids_from(3)) == []

    def test_the_fingerprint_follows_the_content(self, tmp_path: Path) -> None:
        first, second = tmp_path / "1.tsv", tmp_path / "2.tsv"
        write_sitemap_decks(first, [SitemapDeck(_A, "")])
        write_sitemap_decks(second, [SitemapDeck(_B, "")])
        assert (
            MoxfieldPlanFile(first).fingerprint != MoxfieldPlanFile(second).fingerprint
        )
        assert len(MoxfieldPlanFile(first).fingerprint) == 64

    def test_a_malformed_line_raises_when_reached(self, tmp_path: Path) -> None:
        path = tmp_path / "crawl_plan.tsv"
        path.write_text(f"{_A}\t\nbroken\n", encoding="utf-8")
        walk = MoxfieldPlanFile(path).ids_from(0)
        assert next(walk) == (0, _A)
        with pytest.raises(ValueError):
            next(walk)
