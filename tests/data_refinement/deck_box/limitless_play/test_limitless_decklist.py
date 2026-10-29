from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardEntry,
    parse_standings_row,
)
from tests.data_refinement.deck_box.limitless_play.fixtures import raw_entry


class TestParseStandingsRow:
    def test_single_decklist_flattens_sections(self) -> None:
        row = {
            "deck": {"name": "Gardevoir"},
            "decklist": {
                "pokemon": [raw_entry(4, "PAR", "57", "A")],
                "trainer": [raw_entry(2, "SVI", "1", "B")],
                "energy": [],
            },
        }

        (decklist,) = parse_standings_row("t1", 3, row)

        assert decklist.source_id == "t1:3"
        assert decklist.archetype_name == "Gardevoir"
        assert decklist.entries == (
            LimitlessCardEntry(4, "PAR", "57", "A"),
            LimitlessCardEntry(2, "SVI", "1", "B"),
        )
        assert decklist.card_count == 6

    def test_no_decklist_yields_nothing(self) -> None:
        assert parse_standings_row("t1", 0, {"decklist": None, "deck": None}) == []

    def test_multi_deck_row_yields_one_per_deck(self) -> None:
        row = {
            "decklists": {
                "1": {"pokemon": [raw_entry(60, "PAR", "1", "A")]},
                "2": {"pokemon": [raw_entry(60, "PAR", "2", "B")]},
            },
            "decks": {"1": {"name": "One"}, "2": {"name": "Two"}},
        }

        decklists = parse_standings_row("t1", 0, row)

        assert [(d.source_id, d.archetype_name) for d in decklists] == [
            ("t1:0:1", "One"),
            ("t1:0:2", "Two"),
        ]
