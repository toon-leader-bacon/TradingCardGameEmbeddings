import json

import pytest

from src.data_retrieval.mtgtop8.formats import MtgTop8Format
from src.data_retrieval.mtgtop8.listing import ListingRow, parse_search_page
from tests.data_retrieval.mtgtop8.fixtures import PageDeck, search_page_html


class TestParseSearchPage:
    def test_reads_each_row_in_page_order(self) -> None:
        html = search_page_html(
            [
                PageDeck(911101),
                PageDeck(910054, 93472, "EDH", "Slimefoot And Squee", "Cup 109", "5-8"),
            ]
        )

        rows = parse_search_page(html)

        assert rows == [
            ListingRow(
                911101,
                93623,
                MtgTop8Format.PAUPER,
                "Elves",
                "Bournemouth League @ The Trove (Bournemouth, UK)",
                "2",
                "09/11/26",
            ),
            ListingRow(
                910054,
                93472,
                MtgTop8Format.DUEL_COMMANDER,
                "Slimefoot And Squee",
                "Cup 109",
                "5-8",
                "09/11/26",
            ),
        ]

    def test_a_page_past_the_end_lists_none(self) -> None:
        assert parse_search_page(search_page_html([])) == []

    def test_a_page_without_the_search_form_raises(self) -> None:
        with pytest.raises(ValueError, match="search form"):
            parse_search_page("<html><body>Maintenance</body></html>")

    def test_an_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_search_page(search_page_html([PageDeck(1, format_code="XYZ")]))

    def test_a_row_without_its_deck_link_raises(self) -> None:
        html = search_page_html([PageDeck(1)]).replace("&d=1", "")

        with pytest.raises(ValueError, match="deck link"):
            parse_search_page(html)


class TestListingRowLines:
    def test_round_trips(self) -> None:
        row = ListingRow(1, 2, MtgTop8Format.LEGACY, "Æther Vial", "L", "1", "d")

        assert ListingRow.from_line(row.to_line() + "\n") == row

    def test_the_format_is_written_as_its_code(self) -> None:
        row = ListingRow(1, 2, MtgTop8Format.CEDH, "x", "y", "1", "d")

        assert json.loads(row.to_line())["format"] == "cEDH"

    @pytest.mark.parametrize(
        "line",
        [
            "not json",
            "[1, 2]",
            '{"deck_id": 1}',
            '{"deck_id": "1", "event_id": 2, "format": "PAU", "deck_name": "x",'
            ' "event_name": "y", "rank": "1", "date": "d"}',
            '{"deck_id": 1, "event_id": 2, "format": "XYZ", "deck_name": "x",'
            ' "event_name": "y", "rank": "1", "date": "d"}',
        ],
    )
    def test_a_malformed_line_raises(self, line: str) -> None:
        with pytest.raises(ValueError, match="malformed listing line"):
            ListingRow.from_line(line)
