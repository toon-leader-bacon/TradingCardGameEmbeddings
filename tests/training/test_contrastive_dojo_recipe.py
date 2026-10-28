"""ContrastiveDojoRecipe: its dealer comes from the shelf's deck snapshot and
split index, and split caches are named by deck split."""

from pathlib import Path
from typing import Any

import pytest

from src.dojos.file_managers.deck_split_index import deck_split_tag
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.dojo_recipes import contrastive_dojo_recipe
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.dojo_recipes.contrastive_dojo_recipe import ContrastiveDojoRecipe
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext


def _tag(holdout: HoldoutSpec) -> str:
    return deck_split_tag(holdout.deck_split_key(GameId.MTG))


def test_runs_under_different_deck_splits_never_share_a_cache() -> None:
    a = HoldoutSpec(0, (8, 1, 1), frozenset())
    b = HoldoutSpec(1, (8, 1, 1), frozenset())
    assert _tag(a) != _tag(b)


def test_runs_under_the_same_deck_split_share_a_cache() -> None:
    # Card tiers and other held-out games do not change one game's deck split
    a = HoldoutSpec(0, (8, 1, 1), frozenset())
    b = HoldoutSpec(0, (1, 0, 0), frozenset({GameId.GWENT}))
    assert _tag(a) == _tag(b)


class _SnapshotShelf(CardShelf):
    """Records the snapshot and split index requests, handing out tokens."""

    def __init__(self) -> None:
        super().__init__(Path("unused"))
        self.calls: list[tuple[Any, ...]] = []

    def card_binder(self, game: GameId) -> Any:
        return f"binder:{game.value}"

    def deck_snapshot(self, game: GameId, deck_box_path: Path) -> Any:
        self.calls.append(("deck_snapshot", game, deck_box_path))
        return "snapshot"

    def deck_split_index(self, snapshot: Any, holdout: HoldoutSpec) -> Any:
        self.calls.append(("deck_split_index", snapshot, holdout))
        return "split index"


def test_the_dealer_deals_from_the_shelfs_snapshot_and_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dealers: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        contrastive_dojo_recipe,
        "DeckBoxDealer",
        lambda *args: dealers.append(args) or "dealer",
    )
    monkeypatch.setattr(
        contrastive_dojo_recipe, "ContrastiveDojo", lambda *args, **kwargs: "dojo"
    )
    shelf = _SnapshotShelf()
    holdout = HoldoutSpec.no_holdout()
    context = DojoBuildContext(
        shelf=shelf, holdout=holdout, card_embedding_size=8, rng_seed=5
    )

    ContrastiveDojoRecipe(GameId.MTG, Path("x/mtg.db")).build("c.mtg", context)

    assert shelf.calls == [
        ("deck_snapshot", GameId.MTG, Path("x/mtg.db")),
        ("deck_split_index", "snapshot", holdout),
    ]
    assert dealers == [("snapshot", "split index", 5)]
