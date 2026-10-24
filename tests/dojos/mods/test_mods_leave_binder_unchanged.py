"""Every Mod in src/ leaves the CardBinder's cards, and the datum it was
given, exactly as they were.

Each case builds datums from cards read out of a CardBinder, applies its
mod many times, and then compares the binder and the datums with
snapshots taken before. A snapshot is each card's repr, so a key
reordered in place counts as a change too. Card contents nest dicts and
lists so nested path edits are exercised.

test_every_mod_has_a_case imports every module under src/ and fails on a
concrete Mod subclass with no case here, so a new mod gets this check
too.
"""

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

import pytest

import src
from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.multi_group_binary_classification.group_swap_mod import (
    GroupSwapMod,
)
from src.dojos.mods.card_field_mods import (
    FieldMask,
    RandomKeyMaskMod,
    ShuffleKeysMod,
    WeightedFieldMaskMod,
)
from src.dojos.mods.common_mods import MaskTargetKeyMod, NoOpMod, ShuffleDeckMod
from src.dojos.mods.deck_mods import (
    CardDropoutMod,
    CardSubsampleMod,
    DuplicateCollapseMod,
)
from src.dojos.mods.mod import Mod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.per_game_mod import PerGameMod
from src.schema.card import GenericCard, Provenance
from src.schema.card_factory import MissingPathPolicy
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.type_hints import InputShape, TrainingDatum
from src.utils.drop_table import DropTable

# Applications per datum; enough for every random branch of every mod
_ROUNDS = 25

# Datum shapes a mod accepts, named once for the case table
_ALL_SHAPES = (InputShape.SINGLE_CARD, InputShape.MULTI_CARD, InputShape.MULTI_GROUP)
_DECK_SHAPES = (InputShape.MULTI_CARD, InputShape.MULTI_GROUP)


@dataclass(frozen=True)
class ModCase:
    """One mod configuration to check.

    build: makes a fresh mod from a seed. shapes: the datum shapes the
    mod accepts. mod_class: the Mod subclass this case covers, for the
    coverage check.
    """

    case_id: str
    mod_class: type[Mod]
    build: Callable[[int], Mod]
    shapes: tuple[InputShape, ...]


def _nested_mask_table() -> DropTable[FieldMask]:
    """Masks at the top level, inside a dict and inside a list of dicts,
    plus "mask nothing". Inputs: none. Output: DropTable[FieldMask].
    Side effects: none. Exceptions: none."""
    return DropTable.of(
        [
            (1, FieldMask()),
            (1, FieldMask.of_keys("name", "power")),
            (1, FieldMask((("meta", "armor"),))),
            (
                1,
                FieldMask((("card_faces", 0, "oracle_text"), ("legalities", "modern"))),
            ),
            (1, FieldMask((("attacks", 1, "cost"),))),
        ]
    )


def _per_game_mod(seed: int) -> PerGameMod:
    """A PerGameMod with a masking and an augmenting pipeline per game.
    Inputs: seed. Output: PerGameMod. Side effects: none.
    Exceptions: none."""
    return PerGameMod(
        {
            GameId.GWENT: ModPipeline(
                [MaskTargetKeyMod(("meta", "tags")), ShuffleKeysMod(rng_seed=seed)]
            ),
            GameId.MTG: ModPipeline(
                [
                    MaskTargetKeyMod(("card_faces", 1, "name")),
                    RandomKeyMaskMod(rng_seed=seed),
                ]
            ),
        },
        train_only=True,
    )


_CASES = (
    ModCase("no_op", NoOpMod, lambda seed: NoOpMod(), _ALL_SHAPES),
    ModCase(
        "shuffle_keys",
        ShuffleKeysMod,
        lambda seed: ShuffleKeysMod(rng_seed=seed),
        _ALL_SHAPES,
    ),
    ModCase(
        "random_key_mask",
        RandomKeyMaskMod,
        lambda seed: RandomKeyMaskMod(probability=1.0, rng_seed=seed),
        _ALL_SHAPES,
    ),
    ModCase(
        "weighted_field_mask",
        WeightedFieldMaskMod,
        lambda seed: WeightedFieldMaskMod(_nested_mask_table(), rng_seed=seed),
        _ALL_SHAPES,
    ),
    ModCase(
        "shuffle_keys_bounded",
        ShuffleKeysMod,
        lambda seed: ShuffleKeysMod(rng_seed=seed, variants=4),
        _ALL_SHAPES,
    ),
    ModCase(
        "random_key_mask_bounded",
        RandomKeyMaskMod,
        lambda seed: RandomKeyMaskMod(probability=1.0, rng_seed=seed, variants=4),
        _ALL_SHAPES,
    ),
    ModCase(
        "weighted_field_mask_bounded",
        WeightedFieldMaskMod,
        lambda seed: WeightedFieldMaskMod(
            _nested_mask_table(), rng_seed=seed, variants=4
        ),
        _ALL_SHAPES,
    ),
    ModCase(
        "mask_target_top_level",
        MaskTargetKeyMod,
        lambda seed: MaskTargetKeyMod("name"),
        (InputShape.SINGLE_CARD,),
    ),
    ModCase(
        "mask_target_nested",
        MaskTargetKeyMod,
        lambda seed: MaskTargetKeyMod(
            ("attacks", 0, "damage"), missing=MissingPathPolicy.PASS
        ),
        (InputShape.SINGLE_CARD,),
    ),
    ModCase(
        "mask_target_added",
        MaskTargetKeyMod,
        lambda seed: MaskTargetKeyMod("not_on_any_card", missing=MissingPathPolicy.ADD),
        (InputShape.SINGLE_CARD,),
    ),
    ModCase("per_game", PerGameMod, _per_game_mod, (InputShape.SINGLE_CARD,)),
    ModCase(
        "shuffle_deck",
        ShuffleDeckMod,
        lambda seed: ShuffleDeckMod(rng_seed=seed),
        (InputShape.MULTI_CARD,),
    ),
    ModCase(
        "card_dropout",
        CardDropoutMod,
        lambda seed: CardDropoutMod(0.5, frozenset({0, 1}), rng_seed=seed),
        (InputShape.MULTI_GROUP,),
    ),
    ModCase(
        "card_dropout_one_group",
        CardDropoutMod,
        lambda seed: CardDropoutMod(0.5, rng_seed=seed),
        _DECK_SHAPES,
    ),
    ModCase(
        "card_subsample",
        CardSubsampleMod,
        lambda seed: CardSubsampleMod(0.5, frozenset({0, 1}), rng_seed=seed),
        (InputShape.MULTI_GROUP,),
    ),
    ModCase(
        "duplicate_collapse",
        DuplicateCollapseMod,
        lambda seed: DuplicateCollapseMod(frozenset({0, 1}), rng_seed=seed),
        (InputShape.MULTI_GROUP,),
    ),
    ModCase(
        "group_swap",
        GroupSwapMod,
        lambda seed: GroupSwapMod(swap_probability=0.5, rng_seed=seed),
        (InputShape.MULTI_GROUP,),
    ),
)


def _card(game: GameId, name: str, raw_content: dict[str, Any]) -> GenericCard:
    """A card with fixed provenance. Inputs: game, name, raw_content.
    Output: GenericCard. Side effects: none. Exceptions: none."""
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name=name,
        raw_content=raw_content,
        provenance=Provenance(
            DataSource.SYSTEM, "test", datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def _binder() -> CardBinder:
    """A binder of four cards over three games, each nesting dicts and
    lists. Inputs: none. Output: CardBinder. Side effects: none.
    Exceptions: none."""
    binder = CardBinder()
    for card in (
        _card(
            GameId.GWENT,
            "Adalbertus",
            {
                "name": "Adalbertus",
                "faction": "syndicate",
                "power": "6",
                "meta": {"armor": "0", "tags": ["Human", "Blindeyes"]},
            },
        ),
        _card(
            GameId.MTG,
            "Fire // Ice",
            {
                "name": "Fire // Ice",
                "card_faces": [
                    {"name": "Fire", "oracle_text": "Fire deals 2 damage."},
                    {"name": "Ice", "oracle_text": "Tap target permanent."},
                ],
                "legalities": {"modern": "legal", "standard": "not_legal"},
                "colors": ["R", "U"],
            },
        ),
        _card(
            GameId.POKEMON,
            "Pikachu",
            {
                "name": "Pikachu",
                "hp": "60",
                "attacks": [
                    {"name": "Gnaw", "cost": ["Colorless"], "damage": "10"},
                    {"name": "Thunder Jolt", "cost": ["Lightning", "Colorless"]},
                ],
            },
        ),
        _card(GameId.POKEMON, "Empty", {}),
    ):
        binder.create(card)
    return binder


def _binder_cards(binder: CardBinder) -> list[GenericCard]:
    """Every card in binder, read through its public accessors, in a
    fixed order. Inputs: binder. Output: list[GenericCard].
    Side effects: none. Exceptions: none."""
    cards = [card for game in GameId for card in binder.all_cards(game)]
    return sorted(cards, key=lambda card: card.name)


def _snapshot(value: object) -> str:
    """repr of value: every card's fields with raw_content in key order,
    and every list in order. Inputs: value. Output: str.
    Side effects: none. Exceptions: none."""
    return repr(value)


def _datums(cards: list[GenericCard], shape: InputShape) -> list[TrainingDatum[Any]]:
    """Datums of one shape built from cards. Decks hold a duplicate (the
    same card object twice) so a mod meeting one card twice is covered.

    Inputs: cards (at least two), shape. Output: list[TrainingDatum].
    Side effects: none. Exceptions: ValueError for an unsupported shape.
    """
    deck = [*cards, cards[0]]
    if shape is InputShape.SINGLE_CARD:
        return [(card, 1.0) for card in cards]
    if shape is InputShape.MULTI_CARD:
        return [(deck, 1.0), (list(reversed(deck)), 0.0)]
    if shape is InputShape.MULTI_GROUP:
        return [([deck, cards[1:]], 1.0), ([cards[:2], deck], 0.0)]
    raise ValueError(f"no datums for {shape}")


def _case_shapes() -> list[Any]:
    """One pytest param per (case, shape). Inputs: none.
    Output: list of pytest params. Side effects: none. Exceptions: none."""
    return [
        pytest.param(case, shape, id=f"{case.case_id}-{shape.name.lower()}")
        for case in _CASES
        for shape in case.shapes
    ]


@pytest.mark.parametrize("case, shape", _case_shapes())
def test_mod_leaves_binder_and_datums_unchanged(
    case: ModCase, shape: InputShape
) -> None:
    binder = _binder()
    datums = _datums(_binder_cards(binder), shape)
    binder_before = _snapshot(_binder_cards(binder))
    datums_before = _snapshot(datums)

    # Many seeds, both entry points
    for seed in range(_ROUNDS):
        mod = case.build(seed)
        mod.apply(datums)
        for datum in datums:
            mod.apply_single(datum)

    assert _snapshot(_binder_cards(binder)) == binder_before
    assert _snapshot(datums) == datums_before


@pytest.mark.parametrize("case, shape", _case_shapes())
def test_mod_changes_something(case: ModCase, shape: InputShape) -> None:
    # Guards the test above against cases whose mod never fires
    if case.mod_class is NoOpMod:
        pytest.skip("NoOpMod never changes anything")
    datums = _datums(_binder_cards(_binder()), shape)
    before = _snapshot(datums)
    changed = any(
        _snapshot(case.build(seed).apply(datums)) != before for seed in range(_ROUNDS)
    )
    assert changed


def _concrete_mod_classes() -> set[type[Mod]]:
    """Every concrete Mod subclass defined under src/, after importing
    every module there. Inputs: none. Output: set of classes.
    Side effects: imports every src module. Exceptions: ImportError from
    a module that fails to import."""
    for module in pkgutil.walk_packages(src.__path__, "src."):
        importlib.import_module(module.name)
    found: set[type[Mod]] = set()
    pending: list[type[Mod]] = [Mod]
    while pending:
        cls = pending.pop()
        for subclass in cls.__subclasses__():
            pending.append(subclass)
            if subclass.__module__.startswith("src.") and not inspect.isabstract(
                subclass
            ):
                found.add(subclass)
    return found


def test_every_mod_has_a_case() -> None:
    covered = {case.mod_class for case in _CASES}
    missing = sorted(cls.__qualname__ for cls in _concrete_mod_classes() - covered)
    assert not missing, f"add a ModCase for: {missing}"
