from uuid import UUID, uuid4

import pytest

from src.schema.game_id import GameId
from src.schema.holdout import (
    DEFAULT_DECK_SPLIT_RATIOS,
    CardTier,
    HoldoutSpec,
    UnknownDeckSplitError,
)
from src.schema.splits import Split

# Pinned for UUID(int=1)..UUID(int=8) under seed 0 and (1, 1, 1) ratios: a
# change here means every card tier or deck split on disk moved
_PINNED_IDS = [UUID(int=i) for i in range(1, 9)]
_PINNED_DECK_SPLITS = [
    "test",
    "train",
    "train",
    "test",
    "train",
    "test",
    "validation",
    "train",
]
_PINNED_CARD_TIERS = [
    "train",
    "train",
    "validation",
    "train",
    "train",
    "test",
    "train",
    "test",
]


def _spec(ratios=(8.0, 1.0, 1.0), games=frozenset()) -> HoldoutSpec:
    return HoldoutSpec(seed=7, tier_ratios=ratios, held_out_games=games)


def test_tier_is_deterministic() -> None:
    card_id = uuid4()
    assert _spec().tier_of(card_id, GameId.MTG) == _spec().tier_of(card_id, GameId.MTG)


def test_held_out_game_is_always_validation() -> None:
    spec = _spec(ratios=(1, 0, 0), games=frozenset({GameId.GWENT}))
    assert spec.tier_of(uuid4(), GameId.GWENT) == CardTier.VALIDATION
    assert spec.tier_of(uuid4(), GameId.MTG) == CardTier.TRAIN


def test_tier_fractions_are_roughly_respected() -> None:
    spec = _spec(ratios=(7, 2, 1))
    tiers = [spec.tier_of(uuid4(), GameId.MTG) for _ in range(5000)]
    assert 0.15 < tiers.count(CardTier.TEST) / 5000 < 0.25
    assert 0.06 < tiers.count(CardTier.VALIDATION) / 5000 < 0.14


def test_different_seeds_hold_out_different_cards() -> None:
    ids = [uuid4() for _ in range(200)]
    a = HoldoutSpec(1, (7, 3, 0), frozenset())
    b = HoldoutSpec(2, (7, 3, 0), frozenset())
    assert [a.tier_of(i, GameId.MTG) for i in ids] != [
        b.tier_of(i, GameId.MTG) for i in ids
    ]


def test_visibility_is_nested() -> None:
    spec = _spec()
    assert spec.visible_tiers(Split.TRAIN) == {CardTier.TRAIN}
    assert spec.visible_tiers(Split.TEST) == {CardTier.TRAIN, CardTier.TEST}
    assert spec.visible_tiers(Split.VALIDATION) == set(CardTier)


@pytest.mark.parametrize("ratios", [(-1, 1, 1), (0, 0, 0)])
def test_invalid_ratios_rejected(ratios: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        _spec(ratios=ratios)


class TestJsonRoundTrip:
    @pytest.mark.parametrize(
        "spec",
        [
            HoldoutSpec.no_holdout(),
            HoldoutSpec(seed=3, tier_ratios=(8, 1, 1), held_out_games=frozenset()),
            HoldoutSpec(
                seed=-1,
                tier_ratios=(0.7, 0.2, 0.1),
                held_out_games=frozenset({GameId.GWENT, GameId.MTG}),
            ),
        ],
    )
    def test_from_json_inverts_to_json(self, spec: HoldoutSpec) -> None:
        assert HoldoutSpec.from_json(spec.to_json()) == spec

    def test_equal_specs_serialize_identically(self) -> None:
        games_a = frozenset({GameId.GWENT, GameId.MTG})
        games_b = frozenset({GameId.MTG, GameId.GWENT})
        a = HoldoutSpec(seed=1, tier_ratios=(1, 1, 1), held_out_games=games_a)
        b = HoldoutSpec(seed=1, tier_ratios=(1, 1, 1), held_out_games=games_b)
        assert a.to_json() == b.to_json()

    def test_a_round_tripped_spec_assigns_the_same_tiers(self) -> None:
        spec = HoldoutSpec(seed=5, tier_ratios=(8, 1, 1), held_out_games=frozenset())
        copy = HoldoutSpec.from_json(spec.to_json())
        card_ids = [uuid4() for _ in range(200)]
        assert [spec.tier_of(c, GameId.MTG) for c in card_ids] == [
            copy.tier_of(c, GameId.MTG) for c in card_ids
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            "[]",
            '{"seed": 0, "tier_ratios": [1, 0, 0]}',
            '{"seed": 0, "tier_ratios": [1, 0, 0], "held_out_games": [], "x": 1}',
            '{"seed": "0", "tier_ratios": [1, 0, 0], "held_out_games": []}',
            '{"seed": true, "tier_ratios": [1, 0, 0], "held_out_games": []}',
            '{"seed": 0, "tier_ratios": [1, 0], "held_out_games": []}',
            '{"seed": 0, "tier_ratios": [1, "a", 0], "held_out_games": []}',
            '{"seed": 0, "tier_ratios": [0, 0, 0], "held_out_games": []}',
            '{"seed": 0, "tier_ratios": [1, 0, 0], "held_out_games": "mtg"}',
            '{"seed": 0, "tier_ratios": [1, 0, 0], "held_out_games": ["chess"]}',
        ],
    )
    def test_malformed_input_raises_value_error(self, text: str) -> None:
        with pytest.raises(ValueError):
            HoldoutSpec.from_json(text)


class TestDeckSplit:
    def test_split_is_deterministic(self) -> None:
        deck_id = uuid4()
        assert _spec().deck_split_of(deck_id, GameId.MTG) == _spec().deck_split_of(
            deck_id, GameId.MTG
        )

    def test_splits_and_tiers_are_stable_across_processes(self) -> None:
        # Fixed expectations: the hash is sha256, never Python's hash()
        spec = HoldoutSpec(0, (1, 1, 1), frozenset(), deck_split_ratios=(1, 1, 1))
        splits = [spec.deck_split_of(i, GameId.MTG).value for i in _PINNED_IDS]
        tiers = [spec.tier_of(i, GameId.MTG).value for i in _PINNED_IDS]
        assert splits == _PINNED_DECK_SPLITS
        assert tiers == _PINNED_CARD_TIERS

    def test_split_fractions_are_roughly_respected(self) -> None:
        spec = HoldoutSpec(7, (1, 0, 0), frozenset(), deck_split_ratios=(7, 2, 1))
        splits = [spec.deck_split_of(uuid4(), GameId.MTG) for _ in range(5000)]
        assert 0.15 < splits.count(Split.TEST) / 5000 < 0.25
        assert 0.06 < splits.count(Split.VALIDATION) / 5000 < 0.14

    def test_a_held_out_games_decks_are_all_validation(self) -> None:
        spec = HoldoutSpec(
            7, (1, 0, 0), frozenset({GameId.GWENT}), deck_split_ratios=(1, 0, 0)
        )
        assert spec.deck_split_of(uuid4(), GameId.GWENT) == Split.VALIDATION
        assert spec.deck_split_of(uuid4(), GameId.MTG) == Split.TRAIN

    def test_deck_and_card_draws_are_independent(self) -> None:
        # Same uuid, same seed and ratios: the salted deck hash still
        # disagrees with the card hash on some uuids
        spec = HoldoutSpec(7, (1, 1, 0), frozenset(), deck_split_ratios=(1, 1, 0))
        ids = [uuid4() for _ in range(200)]
        card_train = [spec.tier_of(i, GameId.MTG) == CardTier.TRAIN for i in ids]
        deck_train = [spec.deck_split_of(i, GameId.MTG) == Split.TRAIN for i in ids]
        assert card_train != deck_train

    def test_the_default_deck_split_is_eight_one_one(self) -> None:
        assert _spec().deck_split_ratios == DEFAULT_DECK_SPLIT_RATIOS == (8, 1, 1)

    @pytest.mark.parametrize("ratios", [(-1, 1, 1), (0, 0, 0)])
    def test_invalid_deck_ratios_rejected(
        self, ratios: tuple[float, float, float]
    ) -> None:
        with pytest.raises(ValueError, match="deck_split_ratios"):
            HoldoutSpec(7, (1, 0, 0), frozenset(), deck_split_ratios=ratios)

    def test_an_unknown_deck_split_refuses_every_deck(self) -> None:
        spec = HoldoutSpec(7, (1, 0, 0), frozenset({GameId.GWENT}), None)
        with pytest.raises(UnknownDeckSplitError):
            spec.deck_split_of(uuid4(), GameId.MTG)
        with pytest.raises(UnknownDeckSplitError):
            spec.deck_split_of(uuid4(), GameId.GWENT)  # held out: still refused
        with pytest.raises(UnknownDeckSplitError):
            spec.deck_split_key(GameId.MTG)


class TestKeys:
    def test_deck_split_key_example(self) -> None:
        assert HoldoutSpec.no_holdout().deck_split_key(GameId.MTG) == (
            '{"scheme": 1, "seed": 0, "deck_split_fractions": [0.8, 0.1, 0.1]}'
        )

    def test_equal_proportion_deck_ratios_share_a_key(self) -> None:
        a = HoldoutSpec(1, (1, 0, 0), frozenset(), deck_split_ratios=(8, 1, 1))
        b = HoldoutSpec(1, (1, 0, 0), frozenset(), deck_split_ratios=(16.0, 2.0, 2.0))
        assert a.deck_split_key(GameId.MTG) == b.deck_split_key(GameId.MTG)

    def test_deck_split_key_ignores_tiers_and_other_held_out_games(self) -> None:
        a = HoldoutSpec(1, (1, 0, 0), frozenset())
        b = HoldoutSpec(1, (8, 1, 1), frozenset({GameId.GWENT}))
        assert a.deck_split_key(GameId.MTG) == b.deck_split_key(GameId.MTG)

    def test_deck_split_key_changes_with_seed_and_holding_out(self) -> None:
        base = HoldoutSpec(1, (1, 0, 0), frozenset())
        reseeded = HoldoutSpec(2, (1, 0, 0), frozenset())
        held_out = HoldoutSpec(1, (1, 0, 0), frozenset({GameId.MTG}))
        keys = {spec.deck_split_key(GameId.MTG) for spec in (base, reseeded, held_out)}
        assert len(keys) == 3

    def test_card_tier_key_example(self) -> None:
        assert HoldoutSpec.no_holdout().card_tier_key() == (
            '{"seed": 0, "tier_fractions": [1.0, 0.0, 0.0], "held_out_games": []}'
        )

    def test_card_tier_key_ignores_the_deck_split(self) -> None:
        a = HoldoutSpec(1, (8, 1, 1), frozenset(), deck_split_ratios=(1, 0, 0))
        b = HoldoutSpec(1, (16, 2, 2), frozenset(), deck_split_ratios=None)
        assert a.card_tier_key() == b.card_tier_key()

    def test_card_tier_key_changes_with_held_out_games(self) -> None:
        a = HoldoutSpec(1, (8, 1, 1), frozenset())
        b = HoldoutSpec(1, (8, 1, 1), frozenset({GameId.GWENT}))
        assert a.card_tier_key() != b.card_tier_key()


class TestDeckSplitJson:
    @pytest.mark.parametrize(
        "deck_split_ratios", [(8, 1, 1), (1, 0, 0), (0.5, 0.25, 0.25), None]
    )
    def test_round_trips(
        self, deck_split_ratios: tuple[float, float, float] | None
    ) -> None:
        spec = HoldoutSpec(3, (8, 1, 1), frozenset(), deck_split_ratios)
        assert HoldoutSpec.from_json(spec.to_json()) == spec

    def test_json_from_before_deck_splits_loads_as_unknown(self) -> None:
        text = '{"seed": 0, "tier_ratios": [1, 0, 0], "held_out_games": []}'
        assert HoldoutSpec.from_json(text).deck_split_ratios is None

    @pytest.mark.parametrize(
        "deck_split_ratios", ["[1, 0]", '[1, "a", 0]', "[0, 0, 0]", '"811"', "true"]
    )
    def test_malformed_deck_ratios_raise(self, deck_split_ratios: str) -> None:
        text = (
            '{"seed": 0, "tier_ratios": [1, 0, 0], "held_out_games": [], '
            f'"deck_split_ratios": {deck_split_ratios}}}'
        )
        with pytest.raises(ValueError):
            HoldoutSpec.from_json(text)
