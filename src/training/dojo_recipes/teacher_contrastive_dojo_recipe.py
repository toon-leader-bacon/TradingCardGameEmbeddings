"""TeacherContrastiveDojoRecipe: a catalog contrastive dojo dealing one
game's teacher-neighbour pseudo-decks (NeighbourDealer) instead of its
real decks. Staple subsampling is not offered (build_dojos refuses a
threshold on any recipe but ContrastiveDojoRecipe)."""

from dataclasses import dataclass
from pathlib import Path

from src.dojos.contrastive.contrastive_style import ContrastiveStyle
from src.dojos.contrastive.dojo import ContrastiveDojo
from src.dojos.contrastive.styles.single_card import SingleCardStyle
from src.dojos.dojo import Dojo
from src.dojos.file_managers.neighbour_dealer import NeighbourDealer
from src.schema.game_id import GameId
from src.teacher.teacher_settings import TeacherViews
from src.training.dojo_recipes.augmentations import build_augmentation_pipeline
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext
from src.training.teacher_assembly import teachers_for


@dataclass(frozen=True)
class TeacherContrastiveDojoRecipe:
    """A contrastive dojo over one game's teacher neighbours, in one
    contrastive style.

    game: whose binder and teacher. deck_box_path: the game's final deck
    box, whose snapshot the teacher is built from. style: builds the pair
    constructor and loss (default: single cards; see NeighbourDealer for
    how pseudo-deck size meets the style's draw).
    """

    game: GameId
    deck_box_path: Path
    style: ContrastiveStyle = SingleCardStyle()

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build. Loads or builds the game's teacher per
        view of context.teacher under context.holdout (teachers_for; a
        first build writes it under the teacher directory), deals them
        through a NeighbourDealer with context.student, seeded by
        context.rng_seed.
        Mods as ContrastiveDojoRecipe.

        Exceptions: ValueError if context.teacher is None or does not list
            game (checked before any data is loaded), and whatever teachers_for, NeighbourDealer
            and ContrastiveDojo raise.

        Example:
            >>> recipe = TeacherContrastiveDojoRecipe(GameId.MTG, box_path)
            >>> recipe.build("teacher_contrastive.mtg", context).name
            'teacher_contrastive.mtg'
        """
        # Validate inputs: a teacher-sourced dojo needs the teacher views
        views = _require_teacher_views(name, self.game, context)

        # The teachers, then their pseudo-decks
        binder = context.shelf.card_binder(self.game)
        teachers = teachers_for(
            self.game, context.shelf, context.holdout, views, self.deck_box_path
        )
        dealer = NeighbourDealer(
            teachers, context.holdout, context.student, context.rng_seed
        )

        # The style builds the pair constructor and loss as a matched set
        return ContrastiveDojo(
            dealer,
            self.style.pair_constructor(context.rng_seed, None),
            binder,
            context.holdout,
            contrastive_loss=self.style.contrastive_loss(),
            name=name,
            mod_pipeline=build_augmentation_pipeline(name, self.game, context),
        )


def _require_teacher_views(
    name: str, game: GameId, context: DojoBuildContext
) -> TeacherViews:
    """context.teacher, or a ValueError naming dojo name when it is None
    or does not list game.

    Private helper - single caller is TeacherContrastiveDojoRecipe.build.
    Inputs: name, game, context. Output: TeacherViews. Side effects: none.
    Exceptions: ValueError if context.teacher is None or lacks game.
    """
    views = context.teacher
    if views is None:
        raise ValueError(f"{name} needs the run config's teacher: section")
    if game not in views.games:
        raise ValueError(f"{name} needs {game.value} in teacher.games")
    return views
