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
