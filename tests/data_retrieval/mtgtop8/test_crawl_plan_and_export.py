from pathlib import Path

import pytest

from src.data_retrieval.mtgtop8.crawl_plan import (
    MtgTop8PlanFile,
    merge_and_shuffle,
    read_listing_rows,
    write_listing_rows,
)
from src.data_retrieval.mtgtop8.deck_export import read_deck_export
from src.data_retrieval.mtgtop8.formats import MtgTop8Format
from tests.data_retrieval.mtgtop8.fixtures import EXPORT, listing_row


class TestMergeAndShuffle:
    def test_keeps_each_deck_once_first_listing_wins(self, tmp_path: Path) -> None:
        first = tmp_path / "page-00001.jsonl"
        second = tmp_path / "page-00002.jsonl"
        write_listing_rows(first, [listing_row(1), listing_row(2)])
        repeat = listing_row(2, MtgTop8Format.LEGACY)  # pushed down a page
        write_listing_rows(second, [repeat, listing_row(3)])

        rows = merge_and_shuffle([first, second])

        assert sorted(row.deck_id for row in rows) == [1, 2, 3]
        assert next(row for row in rows if row.deck_id == 2).format is (
            MtgTop8Format.PAUPER
        )

    def test_the_order_depends_only_on_the_rows(self, tmp_path: Path) -> None:
        forward = tmp_path / "forward.jsonl"
        backward = tmp_path / "backward.jsonl"
        write_listing_rows(forward, [listing_row(n) for n in range(20)])
        write_listing_rows(backward, [listing_row(n) for n in reversed(range(20))])

        assert merge_and_shuffle([forward]) == merge_and_shuffle([backward])

    def test_no_rows_raise(self, tmp_path: Path) -> None:
        empty = tmp_path / "page-00001.jsonl"
        write_listing_rows(empty, [])

        with pytest.raises(ValueError, match="no decks"):
            merge_and_shuffle([empty])


class TestMtgTop8PlanFile:
    def test_walks_and_looks_up_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "crawl_plan.jsonl"
        write_listing_rows(path, [listing_row(7), listing_row(3), listing_row(5)])
        plan = MtgTop8PlanFile(path)

        assert plan.length == 3
        assert list(plan.ids_from(1)) == [(1, 3), (2, 5)]
        assert list(plan.ids_from(3)) == []
        assert plan.row_for(5) == listing_row(5)
        assert len(plan.fingerprint) == 64
        with pytest.raises(KeyError):
            plan.row_for(4)

    def test_read_listing_rows_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "page.jsonl"
        path.write_text(listing_row(1).to_line() + "\n\n", encoding="utf-8")

        assert read_listing_rows(path) == [listing_row(1)]


class TestReadDeckExport:
    def test_an_export_comes_back_unchanged(self) -> None:
        assert read_deck_export(EXPORT) == EXPORT

    @pytest.mark.parametrize("text", ["", "  \r\n"])
    def test_an_empty_answer_is_none(self, text: str) -> None:
        assert read_deck_export(text) is None

    def test_blank_lines_are_allowed(self) -> None:
        text = "4 Lightning Bolt\r\n\r\nSideboard\r\n"

        assert read_deck_export(text) == text

    @pytest.mark.parametrize(
        "text", ["<html><body>Error</body></html>", "4 Bolt\r\nSide board\r\n", "Bolt"]
    )
    def test_anything_else_raises(self, text: str) -> None:
        with pytest.raises(ValueError, match="not an MTGO deck export"):
            read_deck_export(text)
