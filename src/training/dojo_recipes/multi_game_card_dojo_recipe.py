"""MultiGameCardDojoRecipe: a catalog dojo over several games' cards."""

from dataclasses import dataclass

from src.dojos.dojo import Dojo
from src.dojos.mods.per_game_mod import PerGameMod
from src.schema.game_id import GameId
from src.training.dojo_recipes.augmentations import build_augmentation_pipeline
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext
from src.training.dojo_recipes.dojo_constructors import MultiGameCardDojoConstructor


@dataclass(frozen=True)
class MultiGameCardDojoRecipe:
    """A dojo over the cards of several games, through one
    MultiGameCardLookup. Augmentations are each game's own defaults, run
    per card by a PerGameMod; a mod_overrides entry for the dojo replaces
    them for every game."""

    games: tuple[GameId, ...]
    dojo_class: MultiGameCardDojoConstructor

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build.

        Inputs: name, context.
        Output: the dojo, with a PerGameMod of every game's augmentations
            appended after its own task mods.
        Side effects: loads each game's binder through context.shelf; may
            write split files.
        Exceptions: as DojoRecipe.build.
        """
        dojo = self.dojo_class(
            context.shelf.card_lookup(self.games),
            context.holdout,
            context.card_embedding_size,
            name=name,
            rng_seed=context.rng_seed,
        )

        # One augmentation pipeline per game, dispatched by each card's game
        pipelines = {
            game: build_augmentation_pipeline(name, game, context)
            for game in self.games
        }
        if any(pipeline.mods for pipeline in pipelines.values()):
            dojo.append_mods([PerGameMod(pipelines, train_only=True)])
        return dojo
