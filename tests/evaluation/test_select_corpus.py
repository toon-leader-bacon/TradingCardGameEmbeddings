import pytest

from src.evaluation.select_corpus import CorpusSpec, TierFilter, select_corpus
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from tests.evaluation.fakes import FakeCardLookup, make_card

_MTG = [make_card(f"m{i}") for i in range(40)]
_GWENT = [make_card(f"g{i}", GameId.GWENT) for i in range(5)]
_LOOKUP = FakeCardLookup({GameId.MTG: _MTG, GameId.GWENT: _GWENT})


def test_every_card_of_the_chosen_games_in_game_value_order() -> None:
    spec = CorpusSpec(frozenset({GameId.MTG, GameId.GWENT}), tier_filter=None)
    names = [card.name for card in select_corpus(_LOOKUP, spec)]
    # "gwent" < "mtg": the order is fixed, whatever the frozenset's order
    assert names == [card.name for card in _GWENT + _MTG]


def test_only_the_chosen_games() -> None:
    spec = CorpusSpec(frozenset({GameId.GWENT}), tier_filter=None)
    assert [card.name for card in select_corpus(_LOOKUP, spec)] == [
        card.name for card in _GWENT
    ]


def test_a_tier_filter_keeps_exactly_its_tiers() -> None:
    holdout = HoldoutSpec(seed=3, tier_ratios=(1, 1, 1), held_out_games=frozenset())
    only_validation = TierFilter(holdout, frozenset({CardTier.VALIDATION}))
    spec = CorpusSpec(frozenset({GameId.MTG}), tier_filter=only_validation)

    selected = list(select_corpus(_LOOKUP, spec))

    expected = [
        card
        for card in _MTG
        if holdout.tier_of(card.nocab_uuid, card.source_game) is CardTier.VALIDATION
    ]
    assert selected == expected
    assert 0 < len(selected) < len(_MTG)


def test_a_held_out_game_is_entirely_validation() -> None:
    holdout = HoldoutSpec(
        seed=0, tier_ratios=(1, 0, 0), held_out_games=frozenset({GameId.GWENT})
    )
    tier_filter = TierFilter(holdout, frozenset({CardTier.VALIDATION}))
    spec = CorpusSpec(frozenset({GameId.MTG, GameId.GWENT}), tier_filter)
    assert list(select_corpus(_LOOKUP, spec)) == _GWENT


def test_selection_is_lazy() -> None:
    class ExplodingLookup(FakeCardLookup):
        def all_cards(self, source_game: GameId) -> list:  # type: ignore[override]
            raise AssertionError("read before iteration")

    select_corpus(ExplodingLookup({}), CorpusSpec(frozenset({GameId.MTG}), None))


def test_empty_games_or_tiers_are_rejected() -> None:
    with pytest.raises(ValueError):
        CorpusSpec(frozenset(), tier_filter=None)
    with pytest.raises(ValueError):
        TierFilter(HoldoutSpec.no_holdout(), frozenset())
