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
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.mod_specs import (
    RandomKeyMaskSpec,
    ShuffleKeysSpec,
    WeightedFieldMaskSpec,
)
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
