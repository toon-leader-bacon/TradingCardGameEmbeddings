import copy
import logging
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.dojos.mods.card_field_mods import (
    CardFieldMod,
    FieldMask,
    RandomKeyMaskMod,
    ShuffleKeysMod,
    WeightedFieldMaskMod,
)
from src.dojos.mods.mod import MASK_TOKEN, ModTally
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
    def modified_card(self, card: GenericCard) -> GenericCard:
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
            shuffled = mod.modified_card(card)
            assert shuffled.raw_content == card.raw_content
            orders.add(tuple(shuffled.raw_content))
        assert len(orders) > 1
        assert list(card.raw_content) == ["name", "faction", "power", "meta"]

    def test_counts_only_real_reorders_as_changes(self) -> None:
        mod = ShuffleKeysMod(rng_seed=0)
        single = _card({"name": "x"})
        assert mod.modified_card(single) is single
        mod.apply_single((single, 0))
        assert mod.tally.cards_changed == 0

    def test_empty_card_is_unchanged(self) -> None:
        empty = _card({})
        assert ShuffleKeysMod(rng_seed=0).modified_card(empty) is empty


class TestRandomKeyMaskMod:
    @pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
    def test_rejects_bad_probability(self, probability: float) -> None:
        with pytest.raises(ValueError):
            RandomKeyMaskMod(probability=probability)

    def test_masks_exactly_one_top_level_key(self) -> None:
        card = _gwent_card()
        masked = RandomKeyMaskMod(rng_seed=0).modified_card(card)
        changed = [k for k in card.raw_content if masked.raw_content[k] == MASK_TOKEN]
        assert len(changed) == 1

    def test_spreads_over_every_key(self) -> None:
        mod = RandomKeyMaskMod(rng_seed=0)
        card = _gwent_card()
        hits = Counter(
            next(
                k
                for k, v in mod.modified_card(card).raw_content.items()
                if v == MASK_TOKEN
            )
            for _ in range(400)
        )
        assert set(hits) == set(card.raw_content)

    def test_probability_zero_never_masks(self) -> None:
        card = _gwent_card()
        mod = RandomKeyMaskMod(probability=0.0, rng_seed=0)
        assert all(mod.modified_card(card) is card for _ in range(50))

    def test_empty_card_is_unchanged(self) -> None:
        empty = _card({})
        assert RandomKeyMaskMod(rng_seed=0).modified_card(empty) is empty


class TestWeightedFieldMaskMod:
    def test_rejects_a_table_with_a_non_mask_outcome(self) -> None:
        nested = DropTable.of([(1, "faction")])
        table = DropTable.of([(1, FieldMask()), (1, nested)])
        with pytest.raises(TypeError, match="FieldMask"):
            WeightedFieldMaskMod(table)  # type: ignore[arg-type]

    def test_masks_what_the_table_pulls(self) -> None:
        table = DropTable.of([(1, FieldMask.of_keys("faction"))])
        masked = WeightedFieldMaskMod(table, rng_seed=0).modified_card(_gwent_card())
        assert masked.raw_content["faction"] == MASK_TOKEN

    def test_an_empty_mask_row_means_nothing(self) -> None:
        card = _gwent_card()
        table = DropTable.of([(1, FieldMask())])
        assert WeightedFieldMaskMod(table, rng_seed=0).modified_card(card) is card

    def test_weight_for_missing_fields_moves_to_present_ones(self) -> None:
        # back_face is absent, so every pull must land on faction
        table = DropTable.of(
            [(9, FieldMask.of_keys("back_face")), (1, FieldMask.of_keys("faction"))]
        )
        mod = WeightedFieldMaskMod(table, rng_seed=0)
        card = _gwent_card()
        for _ in range(20):
            assert mod.modified_card(card).raw_content["faction"] == MASK_TOKEN

    def test_a_partly_present_mask_masks_what_exists(self) -> None:
        table = DropTable.of([(1, FieldMask.of_keys("faction", "faction-duo"))])
        masked = WeightedFieldMaskMod(table, rng_seed=0).modified_card(_gwent_card())
        assert masked.raw_content["faction"] == MASK_TOKEN
        assert "faction-duo" not in masked.raw_content

    def test_no_fitting_mask_leaves_the_card(self) -> None:
        card = _gwent_card()
        table = DropTable.of([(1, FieldMask.of_keys("back_face"))])
        mod = WeightedFieldMaskMod(table, rng_seed=0)
        assert mod.modified_card(card) is card
        mod.apply_single((card, 0))
        assert mod.tally.cards_failed == 0 and mod.tally.cards_changed == 0

    def test_rolls_into_a_nested_sub_table(self) -> None:
        meta = DropTable.of(
            [(1, FieldMask((("meta", "armor"),))), (1, FieldMask.of_keys("meta"))]
        )
        table = DropTable.of([(1, meta)])
        mod = WeightedFieldMaskMod(table, rng_seed=0)
        results = {
            str(mod.modified_card(_gwent_card()).raw_content["meta"]) for _ in range(40)
        }
        assert results == {
            MASK_TOKEN,
            str({"armor": MASK_TOKEN, "tags": ["Human", "Blindeyes"]}),
        }
