"""The no-mutation rule: every stock mod leaves its input datum unchanged."""

import copy
import random
from datetime import datetime, timezone
from typing import Iterator
from uuid import uuid4

import pytest

from src.dojos.mods.common_mods import MaskTargetKeyMod, ShuffleDeckMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.card import GenericCard, Provenance
from src.schema.card_factory import MissingPathPolicy
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


@pytest.fixture(autouse=True)
def _restore_global_random() -> Iterator[None]:
    """ShuffleDeckMod uses the global `random`; reseeding it here must not
    leak into other tests."""
    state = random.getstate()
    yield
    random.setstate(state)


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


def test_mask_target_key_masks_the_returned_card_only() -> None:
    card = _card({"faction": "monsters", "power": 5})
    snapshot = copy.deepcopy(card.raw_content)

    masked, label = MaskTargetKeyMod("faction").apply_single((card, 3))

    assert isinstance(masked, GenericCard)
    assert masked.raw_content == {"faction": "[MASK]", "power": 5}
    assert label == 3
    assert card.raw_content == snapshot


def test_mask_target_key_adds_missing_top_level_key_by_default() -> None:
    card = _card({"faction": "monsters"})

    masked, _ = MaskTargetKeyMod("faction-duo").apply_single((card, 0))

    assert isinstance(masked, GenericCard)
    assert masked.raw_content["faction-duo"] == "[MASK]"
    assert "faction-duo" not in card.raw_content


def test_mask_target_key_pass_leaves_a_card_without_the_field_alone() -> None:
    card = _card({"faction": "monsters"})
    mod = MaskTargetKeyMod(("faces", 0, "text"), missing=MissingPathPolicy.PASS)

    masked, _ = mod.apply_single((card, 0))

    assert masked is card


def test_mask_target_key_strict_raises_on_a_missing_field() -> None:
    mod = MaskTargetKeyMod("missing", missing=MissingPathPolicy.STRICT)

    with pytest.raises(KeyError):
        mod.apply_single((_card({}), 0))


def test_mask_target_key_accepts_a_nested_field_path() -> None:
    card = _card({"faces": [{"text": "front"}, {"text": "back"}]})

    masked, _ = MaskTargetKeyMod(("faces", 1, "text")).apply_single((card, 0))

    assert isinstance(masked, GenericCard)
    assert masked.raw_content["faces"] == [{"text": "front"}, {"text": "[MASK]"}]
    assert card.raw_content["faces"][1]["text"] == "back"


@pytest.mark.parametrize("key", ["faction", ("faces", 0, "text")])
def test_mask_target_key_keeps_key_attribute_as_given(key: str | tuple) -> None:
    assert MaskTargetKeyMod(key).key == key


def test_mask_target_key_apply_masks_every_datum() -> None:
    cards = [_card({"faction": f"f{i}"}) for i in range(3)]

    out = MaskTargetKeyMod("faction").apply([(card, i) for i, card in enumerate(cards)])

    assert [c.raw_content["faction"] for c, _ in out] == ["[MASK]"] * 3  # type: ignore[union-attr]
    assert [c.raw_content["faction"] for c in cards] == ["f0", "f1", "f2"]


def test_shuffle_deck_returns_new_list_and_leaves_input_order_unchanged() -> None:
    deck = [_card({"i": i}) for i in range(5)]
    original = list(deck)

    shuffled, label = ShuffleDeckMod().apply_single((deck, 7))

    assert shuffled is not deck  # a new list, even if the order happened to match
    assert deck == original
    assert all(a is b for a, b in zip(deck, original))
    assert label == 7


def test_shuffle_deck_is_a_permutation_sharing_the_same_card_objects() -> None:
    deck = [_card({"i": i}) for i in range(8)]
    random.seed(0)

    shuffled, _ = ShuffleDeckMod().apply_single((deck, 0))

    assert isinstance(shuffled, list)
    assert sorted(map(id, shuffled)) == sorted(map(id, deck))


def test_shuffle_deck_is_reproducible_under_the_global_seed() -> None:
    deck = [_card({"i": i}) for i in range(8)]

    random.seed(42)
    first, _ = ShuffleDeckMod().apply_single((deck, 0))
    random.seed(42)
    second, _ = ShuffleDeckMod().apply_single((deck, 0))

    assert first == second


def test_shuffle_deck_of_an_empty_deck_returns_a_new_empty_list() -> None:
    deck: list = []

    shuffled, _ = ShuffleDeckMod().apply_single((deck, 0))

    assert shuffled == [] and shuffled is not deck


def test_mods_reject_the_wrong_input_shape() -> None:
    card = _card({"faction": "x"})

    with pytest.raises(AssertionError):
        MaskTargetKeyMod("faction").apply_single(([card, card], 0))
    with pytest.raises(AssertionError):
        ShuffleDeckMod().apply_single((card, 0))


def test_pipeline_of_several_masks_never_mutates_inputs() -> None:
    cards = [_card({"faction": "monsters", "faction-duo": "monsters_nilfgaard"})]
    data = [(card, 0) for card in cards]
    snapshots = [copy.deepcopy(card.raw_content) for card in cards]
    pipeline = ModPipeline(
        [
            MaskTargetKeyMod("faction", train_only=False),
            MaskTargetKeyMod("faction-duo", train_only=False),
        ]
    )

    out = pipeline.apply(data, is_training=False)

    assert out[0][0].raw_content == {  # type: ignore[union-attr]
        "faction": "[MASK]",
        "faction-duo": "[MASK]",
    }
    assert [card.raw_content for card in cards] == snapshots
