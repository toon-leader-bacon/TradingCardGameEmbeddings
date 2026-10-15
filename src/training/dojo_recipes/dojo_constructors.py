"""The constructor shapes a catalog recipe can build from, and the one
place the catalog tells regression dojos apart (bind_regression_objective)."""

import functools
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar, cast

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_lookup import DeckLookup
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.generic.multi_group_regression.dojo import MultiGroupRegressionDojo
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext


# The catalog holds dojos of every label type, so its constructor shapes
# return GenericDojo[Any]; each dojo class fixes its own label type.
class CardDojoConstructor(Protocol):
    """A dojo class built from a CardBinder alone (every gwent_one and
    dominiontabs dojo, every sts_gg and isotropic card-level dojo)."""

    def __call__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        *,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> GenericDojo[Any]: ...


class DeckDojoConstructor(Protocol):
    """A dojo class that also needs the DeckBox its metric was built from
    (every sts_gg and isotropic deck-level dojo, play_gwent's leader
    mask)."""

    def __call__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckLookup,
        card_embedding_size: int,
        *,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> GenericDojo[Any]: ...


class MultiGameCardDojoConstructor(Protocol):
    """A dojo class built from one CardLookup spanning several games (the
    cross-game rarity dojo)."""

    def __call__(
        self,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        *,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> GenericDojo[Any]: ...


REGRESSION_CELLS = (
    SingleCardRegressionDojo,
    MultiCardRegressionDojo,
    MultiGroupRegressionDojo,
)
_ConstructorT = TypeVar("_ConstructorT")


def bind_regression_objective(
    dojo_class: _ConstructorT, context: DojoBuildContext
) -> _ConstructorT:
    """dojo_class, bound to the run's regression objective if it is a
    regression dojo.

    Inputs: dojo_class (a dojo class, typed as its constructor Protocol),
        context (carries regression_objective).
    Output: for a subclass of a regression cell, a functools.partial with
        objective=context.regression_objective; any other class unchanged.
        (Only regression constructors accept objective, so this is the one
        place the catalog tells them apart.)
    Side effects: none. Exceptions: none.
    """
    if isinstance(dojo_class, type) and issubclass(dojo_class, REGRESSION_CELLS):
        constructor = cast(Callable[..., GenericDojo[Any]], dojo_class)
        return cast(
            _ConstructorT,
            functools.partial(constructor, objective=context.regression_objective),
        )
    return dojo_class
