"""ContrastiveDojoRecipe's split index naming: one file per deck split."""

from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.dojo_recipes.contrastive_dojo_recipe import (
    _contrastive_index_path,
    _deck_split_tag,
)


def _path(holdout: HoldoutSpec) -> str:
    tag = _deck_split_tag(holdout.deck_split_key(GameId.MTG))
    return _contrastive_index_path("mtg.contrastive", tag).name


def test_runs_under_different_deck_splits_never_share_an_index() -> None:
    a = HoldoutSpec(0, (8, 1, 1), frozenset())
    b = HoldoutSpec(1, (8, 1, 1), frozenset())
    assert _path(a) != _path(b)


def test_runs_under_the_same_deck_split_share_an_index() -> None:
    # Card tiers and other held-out games do not change one game's deck split
    a = HoldoutSpec(0, (8, 1, 1), frozenset())
    b = HoldoutSpec(0, (1, 0, 0), frozenset({GameId.GWENT}))
    assert _path(a) == _path(b)
    assert _path(a).startswith("mtg.contrastive.") and _path(a).endswith(".db")
