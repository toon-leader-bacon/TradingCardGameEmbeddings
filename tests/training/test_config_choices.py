"""choice_of and ConfigSection.required_choice_list: the one rule every
config section reads an enum choice by."""

import pytest

from src.schema.game_id import GameId
from src.schema.splits import Split
from src.training.config_parsing.config_section import ConfigSection, choice_of


class TestChoiceOf:
    def test_reads_a_str_enum_member(self) -> None:
        assert choice_of("train", Split, "here") == Split.TRAIN

    def test_reads_a_str_and_enum_member(self) -> None:
        assert choice_of("mtg", GameId, "here") == GameId.MTG

    def test_unknown_text_names_location_and_choices(self) -> None:
        with pytest.raises(ValueError, match=r"here is 'later', want one of \["):
            choice_of("later", Split, "here")


class TestRequiredChoiceList:
    def test_reads_items_in_order(self) -> None:
        section = ConfigSection({"games": ["mtg", "gwent"]}, "config")

        assert section.required_choice_list("games", GameId) == (
            GameId.MTG,
            GameId.GWENT,
        )

    def test_unknown_item_names_its_index(self) -> None:
        section = ConfigSection({"games": ["mtg", "chess"]}, "config")

        with pytest.raises(ValueError, match=r"config\.games\.1 is 'chess'"):
            section.required_choice_list("games", GameId)

    def test_non_string_item_raises(self) -> None:
        section = ConfigSection({"games": ["mtg", 3]}, "config")

        with pytest.raises(ValueError, match=r"config\.games\.1"):
            section.required_choice_list("games", GameId)

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="games"):
            ConfigSection({}, "config").required_choice_list("games", GameId)
