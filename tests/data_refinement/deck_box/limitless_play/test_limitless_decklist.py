import pytest

from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardEntry,
    LimitlessCardRef,
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
            LimitlessCardEntry(4, LimitlessCardRef("PAR", "57", "A")),
            LimitlessCardEntry(2, LimitlessCardRef("SVI", "1", "B")),
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

    def test_a_single_card_section_is_one_copy_and_the_side_deck_is_dropped(
        self,
    ) -> None:
        row = {
            "decklist": {
                "leader": {"name": "Luffy", "set": "OP13", "number": "001"},
                "character": [raw_entry(4, "OP13", "016", "Garp")],
                "side": [raw_entry(2, "OP01", "002", "Sided")],
            }
        }

        (decklist,) = parse_standings_row("t1", 0, row)

        assert decklist.entries == (
            LimitlessCardEntry(1, LimitlessCardRef("OP13", "001", "Luffy")),
            LimitlessCardEntry(4, LimitlessCardRef("OP13", "016", "Garp")),
        )

    def test_a_section_that_is_neither_list_nor_entry_raises(self) -> None:
        with pytest.raises(TypeError):
            parse_standings_row("t1", 0, {"decklist": {"leader": "Luffy"}})

    def test_a_list_entry_without_a_count_raises(self) -> None:
        row = {"decklist": {"character": [{"name": "A", "set": "S", "number": "1"}]}}
        with pytest.raises(KeyError):
            parse_standings_row("t1", 0, row)
