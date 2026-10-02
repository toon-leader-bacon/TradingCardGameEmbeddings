from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.dojos.generic.multi_group_binary_classification.group_swap_mod import (
    GroupSwapMod,
)
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.DOMINION,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.DOMINIONTABS,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestApplySingle:
    def test_always_swaps_at_probability_one(self) -> None:
        a, b = [_card("A")], [_card("B")]

        groups, label = GroupSwapMod(1.0).apply_single(([a, b], 1.0))

        assert groups == [b, a]
        assert groups[0] is b and groups[1] is a
        assert label == 0.0

    def test_never_swaps_at_probability_zero(self) -> None:
        datum = ([[_card("A")], [_card("B")]], 0.0)

        assert GroupSwapMod(0.0).apply_single(datum) is datum

    def test_input_is_not_mutated(self) -> None:
        a, b = [_card("A")], [_card("B")]
        groups = [a, b]

        GroupSwapMod(1.0).apply_single((groups, 1.0))

        assert groups[0] is a and groups[1] is b

    def test_swaps_about_half_with_a_seed(self) -> None:
        mod = GroupSwapMod(0.5, rng_seed=0)
        datum = ([[_card("A")], [_card("B")]], 1.0)

        flipped = sum(mod.apply_single(datum)[1] == 0.0 for _ in range(1000))

        assert 400 < flipped < 600

    def test_same_seed_same_choices(self) -> None:
        datum = ([[_card("A")], [_card("B")]], 1.0)
        mod_1, mod_2 = GroupSwapMod(0.5, rng_seed=7), GroupSwapMod(0.5, rng_seed=7)
        first = [mod_1.apply_single(datum)[1] for _ in range(20)]
        second = [mod_2.apply_single(datum)[1] for _ in range(20)]

        assert first == second

    def test_rejects_empty_group(self) -> None:
        with pytest.raises(ValueError):
            GroupSwapMod(1.0).apply_single(([[_card("A")], []], 1.0))

    def test_rejects_non_float_label(self) -> None:
        with pytest.raises(TypeError):
            GroupSwapMod(1.0).apply_single(([[_card("A")], [_card("B")]], 1))

    def test_rejects_single_card_input(self) -> None:
        with pytest.raises(TypeError):
            GroupSwapMod(1.0).apply_single((_card("A"), 1.0))


class TestConstruction:
    @pytest.mark.parametrize("probability", [-0.1, 1.5])
    def test_rejects_probability_outside_unit_interval(
        self, probability: float
    ) -> None:
        with pytest.raises(ValueError):
            GroupSwapMod(probability)

    def test_is_train_only_and_skipped_off_train(self) -> None:
        datum = ([[_card("A")], [_card("B")]], 1.0)
        pipeline = ModPipeline([GroupSwapMod(1.0)])

        assert GroupSwapMod().train_only
        assert pipeline.apply_single(datum, is_training=False) is datum
