"""TeacherContrastiveDojoRecipe: it deals the run's teacher views through
a NeighbourDealer, and refuses to build without them."""

from pathlib import Path
from typing import Any

import pytest

from src.dojos.file_managers.neighbour_dealer import StudentSettings
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.teacher.teacher_settings import TeacherViews
from src.training.dojo_recipes import teacher_contrastive_dojo_recipe
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext
from src.training.dojo_recipes.teacher_contrastive_dojo_recipe import (
    TeacherContrastiveDojoRecipe,
)
from tests.teacher.teacher_fixtures import settings


class _BinderShelf(CardShelf):
    """Hands out a token binder and records which games were asked for."""

    def __init__(self) -> None:
        super().__init__(Path("unused"))
        self.binder_games: list[GameId] = []

    def card_binder(self, game: GameId) -> Any:
        self.binder_games.append(game)
        return f"binder:{game.value}"


def _views(tmp_path: Path) -> TeacherViews:
    train = settings(tmp_path)
    return TeacherViews({Split.TRAIN: train, Split.TEST: train})


def _context(teacher: TeacherViews | None) -> DojoBuildContext:
    return DojoBuildContext(
        shelf=_BinderShelf(),
        holdout=HoldoutSpec.no_holdout(),
        card_embedding_size=8,
        rng_seed=5,
        teacher=teacher,
        student=StudentSettings(2, 1),
    )


def test_deals_the_views_teachers_through_a_neighbour_dealer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        teacher_contrastive_dojo_recipe,
        "teachers_for",
        lambda *args: calls.append(("teachers_for", args)) or "teachers",
    )
    monkeypatch.setattr(
        teacher_contrastive_dojo_recipe,
        "NeighbourDealer",
        lambda *args: calls.append(("dealer", args)) or "dealer",
    )
    monkeypatch.setattr(
        teacher_contrastive_dojo_recipe,
        "ContrastiveDojo",
        lambda *args, **kwargs: calls.append(("dojo", args)) or "dojo",
    )
    views = _views(tmp_path)
    context = _context(views)

    built = TeacherContrastiveDojoRecipe(GameId.MTG, Path("x/mtg.db")).build(
        "teacher_contrastive.mtg", context
    )

    assert built == "dojo"
    assert calls[0] == (
        "teachers_for",
        (GameId.MTG, context.shelf, context.holdout, views, Path("x/mtg.db")),
    )
    assert calls[1] == (
        "dealer",
        ("teachers", context.holdout, StudentSettings(2, 1), 5),
    )
    assert calls[2][1][0] == "dealer"


def test_without_teacher_views_raises_before_loading(tmp_path: Path) -> None:
    context = _context(None)

    with pytest.raises(ValueError, match="teacher: section"):
        TeacherContrastiveDojoRecipe(GameId.MTG, Path("x/mtg.db")).build(
            "teacher_contrastive.mtg", context
        )
    assert context.shelf.binder_games == []  # type: ignore[attr-defined]


def test_game_without_a_teacher_raises_before_loading(tmp_path: Path) -> None:
    context = _context(_views(tmp_path))

    with pytest.raises(ValueError, match="pokemon"):
        TeacherContrastiveDojoRecipe(GameId.POKEMON, Path("x/p.db")).build(
            "teacher_contrastive.pokemon", context
        )
    assert context.shelf.binder_games == []  # type: ignore[attr-defined]
