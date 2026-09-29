from uuid import uuid4

import pytest

from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from src.schema.splits import Split


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
