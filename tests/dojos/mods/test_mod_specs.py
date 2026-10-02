from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.dojos.augmentation_defaults import (
    DEFAULT_AUGMENTATIONS,
    default_augmentations_for,
)
from src.dojos.mods.card_field_mods import (
    FieldMask,
    RandomKeyMaskMod,
    ShuffleKeysMod,
    WeightedFieldMaskMod,
)
from src.dojos.mods.deck_mods import (
    CardDropoutMod,
    CardSubsampleMod,
    DuplicateCollapseMod,
)
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod import MASK_TOKEN
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.mod_specs import (
    CardDropoutSpec,
    CardSubsampleSpec,
    DeckModSpec,
    DuplicateCollapseSpec,
    RandomKeyMaskSpec,
    ShuffleKeysSpec,
    WeightedFieldMaskSpec,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.utils.drop_table import DropTable

_TABLE = DropTable.of([(1, FieldMask.of_keys("name"))])


class TestSpecs:
    @pytest.mark.parametrize(
        ("spec", "mod_class"),
        [
            (ShuffleKeysSpec(), ShuffleKeysMod),
            (RandomKeyMaskSpec(probability=0.2), RandomKeyMaskMod),
            (WeightedFieldMaskSpec(_TABLE), WeightedFieldMaskMod),
        ],
    )
    def test_build_makes_a_fresh_mod_each_time(
        self, spec: object, mod_class: type
    ) -> None:
        first = spec.build(rng_seed=1)  # type: ignore[attr-defined]
        second = spec.build(rng_seed=1)  # type: ignore[attr-defined]
        assert isinstance(first, mod_class)
        assert first is not second and first.tally is not second.tally

    def test_random_key_mask_spec_validates_at_construction(self) -> None:
        with pytest.raises(ValueError):
            RandomKeyMaskSpec(probability=2.0)

    def test_weighted_spec_validates_its_table_at_construction(self) -> None:
        with pytest.raises(TypeError):
            WeightedFieldMaskSpec(DropTable.of([(1, "name")]))  # type: ignore[arg-type]

    def test_specs_are_immutable(self) -> None:
        spec = RandomKeyMaskSpec(probability=0.2)
        with pytest.raises(AttributeError):
            spec.probability = 0.5  # type: ignore[misc]


class TestDefaults:
    def test_every_game_with_a_contrastive_dojo_has_defaults(self) -> None:
        assert {
            GameId.GWENT,
            GameId.SLAY_THE_SPIRE_2,
            GameId.FLESH_AND_BLOOD,
            GameId.MTG,
            GameId.POKEMON,
            GameId.DOMINION,
        } <= set(DEFAULT_AUGMENTATIONS)

    def test_gwent_defaults(self) -> None:
        names = [
            type(spec).__name__ for spec in default_augmentations_for(GameId.GWENT)
        ]
        assert names == [
            "ShuffleKeysSpec",
            "WeightedFieldMaskSpec",
            "RandomKeyMaskSpec",
        ]

    def test_a_game_without_defaults_gets_none(self) -> None:
        assert default_augmentations_for(GameId.YUGIOH) == ()


class TestPipelineTallies:
    def test_reports_only_mods_that_keep_a_tally_keyed_by_position(self) -> None:
        from src.dojos.mods.common_mods import NoOpMod

        pipeline = ModPipeline([NoOpMod(), ShuffleKeysMod(), ShuffleKeysMod()])
        tallies = pipeline.mod_tallies()
        assert list(tallies) == ["1:ShuffleKeysMod", "2:ShuffleKeysMod"]
        assert tallies["1:ShuffleKeysMod"] is pipeline.mods[1].tally


class TestDeckSpecs:
    @pytest.mark.parametrize(
        ("spec", "mod_class"),
        [
            (CardDropoutSpec(drop_probability=0.1), CardDropoutMod),
            (CardSubsampleSpec(keep_fraction=0.8, groups=(1,)), CardSubsampleMod),
            (DuplicateCollapseSpec(groups=(0, 1)), DuplicateCollapseMod),
        ],
    )
    def test_build_makes_a_fresh_mod_for_its_groups(
        self, spec: DeckModSpec, mod_class: type
    ) -> None:
        first, second = spec.build(rng_seed=1), spec.build(rng_seed=1)
        assert isinstance(first, mod_class)
        assert first is not second and first.tally is not second.tally
        assert first.groups == frozenset(spec.groups)

    def test_groups_default_to_the_deck(self) -> None:
        assert CardDropoutSpec(drop_probability=0.1).groups == (0,)

    @pytest.mark.parametrize(
        "make_spec",
        [
            lambda: CardDropoutSpec(drop_probability=1.5),
            lambda: CardSubsampleSpec(keep_fraction=0.0),
            lambda: DuplicateCollapseSpec(groups=()),
            lambda: DuplicateCollapseSpec(groups=(-1,)),
        ],
    )
    def test_validates_at_construction(self, make_spec) -> None:
        with pytest.raises(ValueError):
            make_spec()


def _rich_card() -> GenericCard:
    """A card carrying most keys the default mask tables name."""
    raw = {
        "name": "Card",
        "rarity": "rare",
        "faction": "monsters",
        "color": "red",
        "set": "abc",
        "types": ["Action"],
        "cost": 3,
        "typebox": "Ninja Action",
        "category": "Beast",
    }
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name="Card",
        raw_content=raw,
        provenance=Provenance(
            DataSource.GWENT_ONE, "id", datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


class TestDefaultsKeepTheTask:
    """Default augmentations appended after a task's own mods (as the
    catalog does for metric dojos) must not undo or reorder the task."""

    @pytest.mark.parametrize("game", sorted(DEFAULT_AUGMENTATIONS, key=str))
    def test_a_masked_target_stays_masked(self, game: GameId) -> None:
        mask = MaskTargetKeyMod("rarity", train_only=False)
        augmentations = [
            spec.build(rng_seed=seed)
            for seed, spec in enumerate(default_augmentations_for(game))
        ]
        pipeline = ModPipeline([mask, *augmentations])
        for _ in range(200):
            card, label = pipeline.apply_single((_rich_card(), "rare"))
            assert card.raw_content["rarity"] == MASK_TOKEN and label == "rare"

    @pytest.mark.parametrize("game", sorted(DEFAULT_AUGMENTATIONS, key=str))
    def test_groups_and_option_order_are_kept(self, game: GameId) -> None:
        options = [_rich_card() for _ in range(4)]
        deck = [_rich_card() for _ in range(6)]
        pipeline = ModPipeline(
            [spec.build(rng_seed=0) for spec in default_augmentations_for(game)]
        )
        for _ in range(50):
            (new_options, new_deck), label = pipeline.apply_single(([options, deck], 2))
            assert label == 2
            assert [c.nocab_uuid for c in new_options] == [
                c.nocab_uuid for c in options
            ]
            assert [c.nocab_uuid for c in new_deck] == [c.nocab_uuid for c in deck]
