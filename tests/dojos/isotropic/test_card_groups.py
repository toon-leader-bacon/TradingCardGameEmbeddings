import pandas as pd

from src.dojos.isotropic.card_groups import CardColumnGroup, DeckColumnGroup
from tests.dojos.isotropic._group_fixtures import HidingLookup, group_world, names


class TestDeckColumnGroup:
    def test_reads_the_named_group_with_copies(self) -> None:
        world = group_world()
        row = pd.Series({"deck": world.add_group(["Witch", "Chapel", "Witch"])})

        cards = DeckColumnGroup(world.box, "deck").cards(row, world.binder)

        assert sorted(names(cards)) == ["Chapel", "Witch", "Witch"]

    def test_distinct_keeps_first_copy_only(self) -> None:
        world = group_world()
        row = pd.Series({"deck": world.add_group(["Witch", "Chapel", "Witch"])})

        cards = DeckColumnGroup(world.box, "deck", distinct=True).cards(
            row, world.binder
        )

        assert names(cards) == ["Witch", "Chapel"]

    def test_always_included_appended_once(self) -> None:
        world = group_world()
        row = pd.Series({"deck": world.add_group(["Witch", "Silver"])})
        group = DeckColumnGroup(
            world.box,
            "deck",
            always_included=(world.uuid_of("Silver"), world.uuid_of("Gold")),
        )

        assert names(group.cards(row, world.binder)) == ["Witch", "Silver", "Gold"]

    def test_missing_or_unparseable_deck_is_empty(self) -> None:
        world = group_world()
        group = DeckColumnGroup(world.box, "deck")

        assert group.cards(pd.Series({"deck": "not-a-uuid"}), world.binder) == []
        missing = "00000000-0000-0000-0000-000000000000"
        assert group.cards(pd.Series({"deck": missing}), world.binder) == []

    def test_hidden_cards_are_dropped(self) -> None:
        world = group_world()
        row = pd.Series({"deck": world.add_group(["Witch", "Chapel"])})
        lookup = HidingLookup(world.binder, {world.uuid_of("Chapel")})

        cards = DeckColumnGroup(world.box, "deck").cards(row, lookup)

        assert names(cards) == ["Witch"]


class TestCardColumnGroup:
    def test_one_card_group(self) -> None:
        world = group_world()
        row = pd.Series({"card": world.uuid("Moat")})

        assert names(CardColumnGroup("card").cards(row, world.binder)) == ["Moat"]

    def test_unknown_card_is_empty(self) -> None:
        world = group_world()
        row = pd.Series({"card": "00000000-0000-0000-0000-000000000000"})

        assert CardColumnGroup("card").cards(row, world.binder) == []
