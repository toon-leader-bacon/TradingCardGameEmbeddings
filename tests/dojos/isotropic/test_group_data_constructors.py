import math

import pandas as pd

from src.dojos.isotropic.card_groups import CardColumnGroup, DeckColumnGroup
from src.dojos.isotropic.group_label_data_constructor import GroupLabelDataConstructor
from src.dojos.isotropic.group_pick_data_constructor import GroupPickDataConstructor
from tests.dojos.isotropic._group_fixtures import HidingLookup, group_world, names


class TestGroupLabelDataConstructor:
    def test_builds_two_groups_and_a_float_label(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {
                "lo": [world.add_group(["Witch", "Copper"])],
                "hi": [world.add_group(["Chapel"])],
                "lo_won": [True],
            }
        )
        constructor = GroupLabelDataConstructor(
            DeckColumnGroup(world.box, "lo"), DeckColumnGroup(world.box, "hi"), "lo_won"
        )

        [(groups, label)] = constructor.build(chunk, world.binder)

        assert [names(group) for group in groups] == [["Witch", "Copper"], ["Chapel"]]
        assert label == 1.0 and isinstance(label, float)

    def test_card_and_kingdom_with_int_label(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {
                "card_uuid": [world.uuid("Moat")],
                "kingdom_uuid": [world.add_group(["Moat", "Witch"])],
                "count": [3],
            }
        )
        constructor = GroupLabelDataConstructor(
            CardColumnGroup("card_uuid"),
            DeckColumnGroup(world.box, "kingdom_uuid"),
            "count",
        )

        [(groups, label)] = constructor.build(chunk, world.binder)

        assert [names(group) for group in groups] == [["Moat"], ["Moat", "Witch"]]
        assert label == 3.0

    def test_skips_rows_with_an_empty_group_or_nan_label(self) -> None:
        world = group_world()
        kingdom = world.add_group(["Moat"])
        chunk = pd.DataFrame(
            {
                "card_uuid": [
                    world.uuid("Moat"),
                    world.uuid("Witch"),
                    world.uuid("Moat"),
                ],
                "kingdom_uuid": [kingdom, kingdom, "not-a-uuid"],
                "label": [1.0, 1.0, 1.0],
            }
        )
        chunk.loc[len(chunk)] = [world.uuid("Moat"), kingdom, math.nan]
        constructor = GroupLabelDataConstructor(
            CardColumnGroup("card_uuid"),
            DeckColumnGroup(world.box, "kingdom_uuid"),
            "label",
        )
        lookup = HidingLookup(world.binder, {world.uuid_of("Witch")})

        assert len(constructor.build(chunk, lookup)) == 1

    def test_label_caster_is_applied(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {"a": [world.add_group(["Moat"])], "b": [world.add_group(["Witch"])]}
        )
        chunk["y"] = [7]
        constructor = GroupLabelDataConstructor(
            DeckColumnGroup(world.box, "a"),
            DeckColumnGroup(world.box, "b"),
            "y",
            label_caster=lambda raw: 0.5,
        )

        assert constructor.build(chunk, world.binder)[0][1] == 0.5


class TestGroupPickDataConstructor:
    def test_one_datum_per_pick_without_context(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {
                "pool": [world.add_group(["Chapel", "Witch", "Moat"])],
                "picks": [[world.uuid("Witch"), world.uuid("Moat")]],
            }
        )
        constructor = GroupPickDataConstructor(
            DeckColumnGroup(world.box, "pool"), "picks"
        )

        result = constructor.build(chunk, world.binder)

        assert [index for _, index in result] == [1, 2]
        assert names(result[0][0]) == ["Chapel", "Witch", "Moat"]

    def test_context_becomes_group_one(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {
                "kingdom": [world.add_group(["Witch", "Moat"])],
                "deck": [world.add_group(["Copper", "Copper", "Estate"])],
                "picks": [[world.uuid("Silver")]],
            }
        )
        constructor = GroupPickDataConstructor(
            DeckColumnGroup(
                world.box,
                "kingdom",
                distinct=True,
                always_included=(world.uuid_of("Copper"), world.uuid_of("Silver")),
            ),
            "picks",
            context=DeckColumnGroup(world.box, "deck"),
        )

        [([options, context], index)] = constructor.build(chunk, world.binder)

        assert names(options) == ["Witch", "Moat", "Copper", "Silver"]
        assert names(context) == ["Copper", "Copper", "Estate"]
        assert index == 3

    def test_picks_outside_the_options_are_dropped(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {
                "pool": [world.add_group(["Chapel", "Witch"])],
                "picks": [[world.uuid("Gold"), "garbage", world.uuid("Chapel")]],
            }
        )
        constructor = GroupPickDataConstructor(
            DeckColumnGroup(world.box, "pool"), "picks"
        )

        assert [index for _, index in constructor.build(chunk, world.binder)] == [0]

    def test_hidden_option_shifts_index_and_hidden_pick_is_dropped(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {
                "pool": [world.add_group(["Chapel", "Witch", "Moat"])],
                "picks": [[world.uuid("Moat"), world.uuid("Chapel")]],
            }
        )
        constructor = GroupPickDataConstructor(
            DeckColumnGroup(world.box, "pool"), "picks"
        )
        lookup = HidingLookup(world.binder, {world.uuid_of("Chapel")})

        [(options, index)] = constructor.build(chunk, lookup)

        assert names(options) == ["Witch", "Moat"] and index == 1

    def test_fewer_than_two_options_skips_the_row(self) -> None:
        world = group_world()
        chunk = pd.DataFrame(
            {"pool": [world.add_group(["Chapel"])], "picks": [[world.uuid("Chapel")]]}
        )
        constructor = GroupPickDataConstructor(
            DeckColumnGroup(world.box, "pool"), "picks"
        )

        assert constructor.build(chunk, world.binder) == []
