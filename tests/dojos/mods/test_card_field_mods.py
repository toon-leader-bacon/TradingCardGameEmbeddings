import copy
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

import pytest

from src.dojos.mods.card_field_mods import (
    UNBOUNDED_VARIANTS,
    CardFieldMod,
    FieldMask,
    RandomKeyMaskMod,
    ShuffleKeysMod,
    WeightedFieldMaskMod,
)
from src.dojos.mods.mod import MASK_TOKEN, ModTally
from src.dojos.mods.variant_seeds import (
    MAX_VARIANTS,
    VARIANT_SEEDS,
    card_variant_rng,
    mod_salt,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.utils.drop_table import DropTable


def _card(raw_content: dict) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name="Test Card",
        raw_content=raw_content,
        provenance=Provenance(
            DataSource.GWENT_ONE, "src-id", datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def _gwent_card() -> GenericCard:
    return _card(
        {
            "name": "Adalbertus",
            "faction": "syndicate",
            "power": "6",
            "meta": {"armor": "0", "tags": ["Human", "Blindeyes"]},
        }
    )


class _ExplodingMod(CardFieldMod):
    def modified_card(self, card: GenericCard, rng: random.Random) -> GenericCard:
        raise RuntimeError("boom")


class TestModTally:
    def test_changed_fraction_is_zero_before_any_card(self) -> None:
        assert ModTally().changed_fraction == 0.0

    def test_record_failure_keeps_the_first_error(self) -> None:
        tally = ModTally()
        assert tally.record_failure(KeyError("a")) is True
        assert tally.record_failure(KeyError("b")) is False
        assert tally.cards_failed == 2
        assert tally.first_failure == repr(KeyError("a"))

    def test_changed_fraction(self) -> None:
        assert ModTally(cards_seen=4, cards_changed=1).changed_fraction == 0.25


class TestFieldMask:
    def test_of_keys_builds_one_step_paths(self) -> None:
        assert FieldMask.of_keys("a", "b").paths == (("a",), ("b",))

    @pytest.mark.parametrize(
        "paths",
        [
            (("costs",), ("costs", "mana")),  # prefix
            (("costs", "mana"), ("costs",)),  # prefix, other order
            (("name",), ("name",)),  # duplicate
        ],
    )
    def test_rejects_overlapping_paths(self, paths: tuple) -> None:
        with pytest.raises(ValueError, match="overlap"):
            FieldMask(paths)

    def test_rejects_a_bare_string_path(self) -> None:
        with pytest.raises(TypeError, match="tuple"):
            FieldMask(("faction",))  # type: ignore[arg-type]

    def test_rejects_malformed_steps(self) -> None:
        with pytest.raises(ValueError):
            FieldMask((("items", -1),))

    def test_sibling_paths_are_fine(self) -> None:
        FieldMask((("meta", "armor"), ("meta", "tags")))

    def test_fits_when_any_path_exists(self) -> None:
        card = _gwent_card()
        assert FieldMask.of_keys("faction").fits(card)
        assert FieldMask((("meta", "tags", 1),)).fits(card)
        assert FieldMask.of_keys("faction", "missing").fits(card)
        assert not FieldMask.of_keys("missing", "absent").fits(card)
        assert not FieldMask((("meta", "tags", 5),)).fits(card)
        assert not FieldMask((("power", "inner"),)).fits(card)  # into a scalar
        assert FieldMask().fits(card)

    def test_applied_to_masks_every_path_and_leaves_the_input(self) -> None:
        card = _gwent_card()
        before = copy.deepcopy(card.raw_content)

        masked = FieldMask((("faction",), ("meta", "armor"))).applied_to(card)

        assert masked.raw_content["faction"] == MASK_TOKEN
        assert masked.raw_content["meta"]["armor"] == MASK_TOKEN
        assert card.raw_content == before

    def test_empty_mask_returns_the_same_card(self) -> None:
        card = _gwent_card()
        assert FieldMask().applied_to(card) is card


class TestCardFieldModShapes:
    def test_maps_over_multi_card_and_grouped_inputs(self) -> None:
        mod = RandomKeyMaskMod(rng_seed=0)
        cards = [_gwent_card(), _gwent_card()]
        groups = [[_gwent_card()], [_gwent_card(), _gwent_card()]]

        multi, _ = mod.apply_single((cards, 1.0))
        grouped, _ = mod.apply_single((groups, 1.0))

        assert isinstance(multi, list) and len(multi) == 2
        assert [len(group) for group in grouped] == [1, 2]  # type: ignore[arg-type]
        assert mod.tally.cards_seen == 5
        assert cards[0].raw_content == _gwent_card().raw_content  # unchanged

    def test_label_passes_through(self) -> None:
        _, label = ShuffleKeysMod(rng_seed=0).apply_single((_gwent_card(), 3.5))
        assert label == 3.5

    def test_apply_maps_every_datum(self) -> None:
        data = [(_gwent_card(), 1), (_gwent_card(), 2)]
        result = ShuffleKeysMod(rng_seed=0).apply(data)
        assert [label for _, label in result] == [1, 2]


class TestBestEffort:
    def test_a_failing_card_passes_through_and_is_counted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mod = _ExplodingMod()
        card = _gwent_card()

        with caplog.at_level(logging.DEBUG):
            first, _ = mod.apply_single((card, 0))
            mod.apply_single((card, 0))

        assert first is card
        assert mod.tally.cards_failed == 2
        assert mod.tally.cards_changed == 0
        assert mod.tally.first_failure is not None
        assert "boom" in mod.tally.first_failure
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # only the first failure warns

    def test_the_mod_is_train_only_by_default(self) -> None:
        assert ShuffleKeysMod().train_only


class TestShuffleKeysMod:
    def test_reorders_keys_without_changing_content(self) -> None:
        card = _gwent_card()
        mod = ShuffleKeysMod(rng_seed=0)
        orders = set()
        for _ in range(30):
            shuffled = mod.modified_card(card, mod.rng)
            assert shuffled.raw_content == card.raw_content
            orders.add(tuple(shuffled.raw_content))
        assert len(orders) > 1
        assert list(card.raw_content) == ["name", "faction", "power", "meta"]

    def test_counts_only_real_reorders_as_changes(self) -> None:
        mod = ShuffleKeysMod(rng_seed=0)
        single = _card({"name": "x"})
        assert mod.modified_card(single, mod.rng) is single
        mod.apply_single((single, 0))
        assert mod.tally.cards_changed == 0

    def test_empty_card_is_unchanged(self) -> None:
        empty = _card({})
        assert (
            ShuffleKeysMod(rng_seed=0).modified_card(empty, random.Random(0)) is empty
        )


class TestRandomKeyMaskMod:
    @pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
    def test_rejects_bad_probability(self, probability: float) -> None:
        with pytest.raises(ValueError):
            RandomKeyMaskMod(probability=probability)

    def test_masks_exactly_one_top_level_key(self) -> None:
        card = _gwent_card()
        masked = RandomKeyMaskMod(rng_seed=0).modified_card(card, random.Random(0))
        changed = [k for k in card.raw_content if masked.raw_content[k] == MASK_TOKEN]
        assert len(changed) == 1

    def test_spreads_over_every_key(self) -> None:
        mod = RandomKeyMaskMod(rng_seed=0)
        card = _gwent_card()
        hits = Counter(
            next(
                k
                for k, v in mod.modified_card(card, mod.rng).raw_content.items()
                if v == MASK_TOKEN
            )
            for _ in range(400)
        )
        assert set(hits) == set(card.raw_content)

    def test_probability_zero_never_masks(self) -> None:
        card = _gwent_card()
        mod = RandomKeyMaskMod(probability=0.0, rng_seed=0)
        assert all(mod.apply_single((card, 0))[0] is card for _ in range(50))

    def test_empty_card_is_unchanged(self) -> None:
        empty = _card({})
        assert (
            RandomKeyMaskMod(rng_seed=0).modified_card(empty, random.Random(0)) is empty
        )


class TestWeightedFieldMaskMod:
    def test_rejects_a_table_with_a_non_mask_outcome(self) -> None:
        nested = DropTable.of([(1, "faction")])
        table = DropTable.of([(1, FieldMask()), (1, nested)])
        with pytest.raises(TypeError, match="FieldMask"):
            WeightedFieldMaskMod(table)  # type: ignore[arg-type]

    def test_masks_what_the_table_pulls(self) -> None:
        table = DropTable.of([(1, FieldMask.of_keys("faction"))])
        masked = WeightedFieldMaskMod(table, rng_seed=0).modified_card(
            _gwent_card(), random.Random(0)
        )
        assert masked.raw_content["faction"] == MASK_TOKEN

    def test_an_empty_mask_row_means_nothing(self) -> None:
        card = _gwent_card()
        table = DropTable.of([(1, FieldMask())])
        assert (
            WeightedFieldMaskMod(table, rng_seed=0).modified_card(
                card, random.Random(0)
            )
            is card
        )

    def test_weight_for_missing_fields_moves_to_present_ones(self) -> None:
        # back_face is absent, so every pull must land on faction
        table = DropTable.of(
            [(9, FieldMask.of_keys("back_face")), (1, FieldMask.of_keys("faction"))]
        )
        mod = WeightedFieldMaskMod(table, rng_seed=0)
        card = _gwent_card()
        for _ in range(20):
            assert mod.modified_card(card, mod.rng).raw_content["faction"] == MASK_TOKEN

    def test_a_partly_present_mask_masks_what_exists(self) -> None:
        table = DropTable.of([(1, FieldMask.of_keys("faction", "faction-duo"))])
        masked = WeightedFieldMaskMod(table, rng_seed=0).modified_card(
            _gwent_card(), random.Random(0)
        )
        assert masked.raw_content["faction"] == MASK_TOKEN
        assert "faction-duo" not in masked.raw_content

    def test_no_fitting_mask_leaves_the_card(self) -> None:
        card = _gwent_card()
        table = DropTable.of([(1, FieldMask.of_keys("back_face"))])
        mod = WeightedFieldMaskMod(table, rng_seed=0)
        assert mod.modified_card(card, mod.rng) is card
        mod.apply_single((card, 0))
        assert mod.tally.cards_failed == 0 and mod.tally.cards_changed == 0

    def test_rolls_into_a_nested_sub_table(self) -> None:
        meta = DropTable.of(
            [(1, FieldMask((("meta", "armor"),))), (1, FieldMask.of_keys("meta"))]
        )
        table = DropTable.of([(1, meta)])
        mod = WeightedFieldMaskMod(table, rng_seed=0)
        results = {
            str(mod.modified_card(_gwent_card(), mod.rng).raw_content["meta"])
            for _ in range(40)
        }
        assert results == {
            MASK_TOKEN,
            str({"armor": MASK_TOKEN, "tags": ["Human", "Blindeyes"]}),
        }


class TestDoNotCacheStamp:
    def test_a_changed_card_is_stamped(self) -> None:
        mod = RandomKeyMaskMod(rng_seed=0)
        masked, _ = mod.apply_single((_gwent_card(), 0))
        assert masked.cacheable is False

    def test_an_unchanged_card_is_not_stamped(self) -> None:
        card = _gwent_card()
        mod = RandomKeyMaskMod(probability=0.0, rng_seed=0)
        passed, _ = mod.apply_single((card, 0))
        assert passed is card and passed.cacheable is True

    def test_every_card_field_mod_is_random_by_default(self) -> None:
        table = DropTable.of([(1, FieldMask.of_keys("faction"))])
        for mod in (ShuffleKeysMod(), RandomKeyMaskMod(), WeightedFieldMaskMod(table)):
            assert mod.deterministic_cards is False


def _text(card: GenericCard) -> str:
    # Key order matters: a reorder is a different text
    return repr(list(card.raw_content.items()))


def _wide_card() -> GenericCard:
    return _card({f"key_{index}": index for index in range(8)})


_FOUR_ROWS = DropTable.of(
    [(1, FieldMask.of_keys(f"key_{index}")) for index in range(4)]
)

_BOUNDED_MODS: dict[str, Callable[[int, int], CardFieldMod]] = {
    "shuffle_keys": lambda seed, n: ShuffleKeysMod(rng_seed=seed, variants=n),
    "random_key_mask": lambda seed, n: RandomKeyMaskMod(rng_seed=seed, variants=n),
    "weighted_field_mask": lambda seed, n: WeightedFieldMaskMod(
        _FOUR_ROWS, rng_seed=seed, variants=n
    ),
}


class TestBoundedVariants:
    @pytest.mark.parametrize("make_mod", _BOUNDED_MODS.values(), ids=_BOUNDED_MODS)
    def test_at_most_n_texts_per_card(
        self, make_mod: Callable[[int, int], CardFieldMod]
    ) -> None:
        card = _wide_card()
        mod = make_mod(0, 3)
        texts = {_text(mod.apply_single((card, 0))[0]) for _ in range(200)}
        assert 1 < len(texts) <= 3

    @pytest.mark.parametrize("make_mod", _BOUNDED_MODS.values(), ids=_BOUNDED_MODS)
    def test_every_run_seed_shares_the_same_variants(
        self, make_mod: Callable[[int, int], CardFieldMod]
    ) -> None:
        card = _wide_card()
        texts = {
            _text(make_mod(seed, 2).apply_single((card, 0))[0])
            for seed in range(20)
            for _ in range(10)
        }
        assert len(texts) <= 2

    def test_one_variant_still_differs_across_cards(self) -> None:
        # The card's uuid is in the seed, so variant 0 is not one draw corpus-wide
        mod = WeightedFieldMaskMod(_FOUR_ROWS, rng_seed=0, variants=1)
        masked_keys = {
            next(
                k
                for k, v in mod.apply_single((_wide_card(), 0))[0].raw_content.items()
                if v == MASK_TOKEN
            )
            for _ in range(40)
        }
        assert len(masked_keys) > 1

    def test_mod_classes_get_different_salts(self) -> None:
        salts = {
            mod_salt(cls)
            for cls in (ShuffleKeysMod, RandomKeyMaskMod, WeightedFieldMaskMod)
        }
        assert len(salts) == 3

    @pytest.mark.parametrize("make_mod", _BOUNDED_MODS.values(), ids=_BOUNDED_MODS)
    def test_bounded_cards_are_not_stamped(
        self, make_mod: Callable[[int, int], CardFieldMod]
    ) -> None:
        mod = make_mod(0, 4)
        results = [mod.apply_single((_wide_card(), 0))[0] for _ in range(20)]
        assert mod.deterministic_cards is True
        assert mod.tally.cards_changed > 0
        assert all(card.cacheable for card in results)

    def test_a_stamped_card_stays_stamped(self) -> None:
        # An unbounded mod upstream makes the text one of many again
        shuffled, _ = ShuffleKeysMod(rng_seed=0).apply_single((_wide_card(), 0))
        bounded = RandomKeyMaskMod(rng_seed=0, variants=4)
        masked, _ = bounded.apply_single((shuffled, 0))
        assert shuffled.cacheable is False
        assert masked is not shuffled and masked.cacheable is False

    @pytest.mark.parametrize("make_mod", _BOUNDED_MODS.values(), ids=_BOUNDED_MODS)
    def test_zero_variants_turns_the_mod_off(
        self, make_mod: Callable[[int, int], CardFieldMod]
    ) -> None:
        card = _wide_card()
        mod = make_mod(0, 0)
        assert all(mod.apply_single((card, 0))[0] is card for _ in range(20))
        assert mod.tally.cards_seen == 20 and mod.tally.cards_changed == 0

    @pytest.mark.parametrize("make_mod", _BOUNDED_MODS.values(), ids=_BOUNDED_MODS)
    def test_negative_variants_draw_fresh_and_stamp(
        self, make_mod: Callable[[int, int], CardFieldMod]
    ) -> None:
        card = _wide_card()
        mod = make_mod(0, UNBOUNDED_VARIANTS)
        results = [mod.apply_single((card, 0))[0] for _ in range(200)]
        changed = [result for result in results if result is not card]
        assert len({_text(result) for result in results}) >= 4
        assert changed and not any(result.cacheable for result in changed)

    @pytest.mark.parametrize("make_mod", _BOUNDED_MODS.values(), ids=_BOUNDED_MODS)
    def test_rejects_more_variants_than_seeds(
        self, make_mod: Callable[[int, int], CardFieldMod]
    ) -> None:
        with pytest.raises(ValueError, match="at most"):
            make_mod(0, MAX_VARIANTS + 1)


class TestVariantSeeds:
    def test_one_hundred_distinct_seeds(self) -> None:
        assert MAX_VARIANTS == 100 == len(set(VARIANT_SEEDS))

    def test_same_inputs_same_draws(self) -> None:
        card_uuid = uuid4()
        first = card_variant_rng(5, card_uuid, 7)
        second = card_variant_rng(5, card_uuid, 7)
        assert [first.random() for _ in range(5)] == [second.random() for _ in range(5)]

    def test_variant_card_and_salt_each_change_the_draws(self) -> None:
        card_uuid = uuid4()
        base = card_variant_rng(5, card_uuid, 7).random()
        assert card_variant_rng(6, card_uuid, 7).random() != base
        assert card_variant_rng(5, uuid4(), 7).random() != base
        assert card_variant_rng(5, card_uuid, 8).random() != base

    @pytest.mark.parametrize("variant", [-1, MAX_VARIANTS])
    def test_rejects_a_variant_outside_the_table(self, variant: int) -> None:
        with pytest.raises(IndexError):
            card_variant_rng(variant, uuid4(), 0)


_GATED_MODS: dict[str, Callable[[float, int], CardFieldMod]] = {
    "shuffle_keys": lambda p, n: ShuffleKeysMod(rng_seed=0, variants=n, probability=p),
    "random_key_mask": lambda p, n: RandomKeyMaskMod(
        probability=p, rng_seed=0, variants=n
    ),
    "weighted_field_mask": lambda p, n: WeightedFieldMaskMod(
        _FOUR_ROWS, rng_seed=0, variants=n, probability=p
    ),
}


class TestProbabilityGate:
    @pytest.mark.parametrize("make_mod", _GATED_MODS.values(), ids=_GATED_MODS)
    @pytest.mark.parametrize("variants", [UNBOUNDED_VARIANTS, 4])
    def test_acts_on_about_probability_of_the_cards(
        self, make_mod: Callable[[float, int], CardFieldMod], variants: int
    ) -> None:
        card = _wide_card()
        mod = make_mod(0.1, variants)
        changed = sum(mod.apply_single((card, 0))[0] is not card for _ in range(2000))
        assert 120 <= changed <= 280  # about 10% of 2,000

    @pytest.mark.parametrize("make_mod", _GATED_MODS.values(), ids=_GATED_MODS)
    def test_zero_probability_never_acts(
        self, make_mod: Callable[[float, int], CardFieldMod]
    ) -> None:
        card = _wide_card()
        mod = make_mod(0.0, 4)
        assert all(mod.apply_single((card, 0))[0] is card for _ in range(50))

    def test_gated_bounded_draws_stay_within_n_texts(self) -> None:
        # The gate is on the mod's own state, so a card that fires still
        # lands on one of its n fixed variants
        card = _wide_card()
        mod = ShuffleKeysMod(rng_seed=0, variants=2, probability=0.5)
        texts = {_text(mod.apply_single((card, 0))[0]) for _ in range(300)}
        assert len(texts) <= 3  # the unchanged card plus at most 2 variants

    @pytest.mark.parametrize("make_mod", _GATED_MODS.values(), ids=_GATED_MODS)
    @pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
    def test_rejects_a_bad_probability(
        self, make_mod: Callable[[float, int], CardFieldMod], probability: float
    ) -> None:
        with pytest.raises(ValueError, match="probability"):
            make_mod(probability, 4)
