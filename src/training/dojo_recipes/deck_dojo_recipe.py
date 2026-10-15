"""DeckDojoRecipe: a catalog dojo built from its game's CardBinder and a
DeckBox."""

from dataclasses import dataclass
from pathlib import Path

from src.dojos.dojo import Dojo
from src.schema.game_id import GameId
from src.training.dojo_recipes.augmentations import append_augmentations
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext
from src.training.dojo_recipes.dojo_constructors import (
    DeckDojoConstructor,
    bind_regression_objective,
)


@dataclass(frozen=True)
class DeckDojoRecipe:
    """A dojo that needs its game's CardBinder and a DeckBox. metric_output,
    when set, replaces the dojo class's own metric file."""

    game: GameId
    dojo_class: DeckDojoConstructor
    deck_box_path: Path
    metric_output: Path | None = None

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build; augmentations as append_augmentations."""
        dojo = bind_regression_objective(self.dojo_class, context)(
            context.shelf.card_binder(self.game),
            context.holdout,
            context.shelf.deck_box(self.deck_box_path),
            context.card_embedding_size,
            path_to_training_data=self.metric_output,
            name=name,
            rng_seed=context.rng_seed,
        )
        return append_augmentations(dojo, name, self.game, context)
