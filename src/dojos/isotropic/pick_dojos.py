"""Per-metric wrappers over the isotropic "which card(s)" metrics: an
option list (plus, for some, a conditioning card group) in, the picked
card's index out, one datum per picked card (GroupPickDataConstructor).

- KingdomOpeningBuyDojo: kingdom + base supply -> the winner's opening
  buy(s) (MultiCardOptionSelectionDojo).
- KingdomVetoDojo: the pre-veto candidate pool -> the vetoed card(s)
  (MultiCardOptionSelectionDojo).
- NextBuyDojo: [kingdom + base supply, partial deck] -> the card(s)
  bought this turn (MultiGroupOptionSelectionDojo).
- NextTrashedCardDojo: [the partial deck's distinct cards, the partial
  deck] -> the card(s) trashed this turn (MultiGroupOptionSelectionDojo).

BASE SUPPLY: a kingdom group lists its kingdom cards, plus Platinum,
Colony and Potion when that game had them, but never the seven piles
every game has. Buys of those (Silver openings, Province buys) are about
a quarter of the labels, so the buy wrappers add BASE_SUPPLY_NAMES to the
options. A pick outside the options (a Black Market buy, ~0.7% of next
buys) is dropped by the constructor.

DECK BOX: every wrapper passes deck_box to its cell as well, so
GenericDojo's version check confirms box and metric share a CardBinder.
"""

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.kingdom_opening_buy_prediction_metric import (  # noqa: E501
    KingdomOpeningBuyPredictionMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_next_buy_metric import (
    NextBuyPredictionMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_next_trashed_card_metric import (  # noqa: E501
    NextTrashedCardMetric,
)
from src.data_refinement.metrics.isotropic.summary.kingdom_veto_prediction_metric import (  # noqa: E501
    KingdomVetoPredictionMetric,
)
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_option_selection.dojo import (
    MultiCardOptionSelectionDojo,
)
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.isotropic.card_groups import DeckColumnGroup
from src.dojos.isotropic.group_pick_data_constructor import GroupPickDataConstructor
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# The seven supply piles every Dominion game has; see module docstring
BASE_SUPPLY_NAMES: tuple[str, ...] = (
    "Copper",
    "Silver",
    "Gold",
    "Estate",
    "Duchy",
    "Province",
    "Curse",
)


def base_supply_uuids(card_binder: CardBinder) -> tuple[UUID, ...]:
    """The nocab_uuids of BASE_SUPPLY_NAMES, in that order.

    Inputs: card_binder (the Dominion binder).
    Output: tuple[UUID, ...], one per name.
    Side effects: none.
    Exceptions: ValueError if a base card is missing from the binder.

    Example:
        >>> len(base_supply_uuids(binder))
        7
    """
    result: list[UUID] = []
    for name in BASE_SUPPLY_NAMES:
        card = card_binder.get_by_name_single(GameId.DOMINION, name)
        if card is None:
            raise ValueError(f"the Dominion card binder has no {name!r}")
        result.append(card.nocab_uuid)
    return tuple(result)


def _kingdom_supply(
    card_binder: CardBinder, deck_box: DeckBox, column: str
) -> DeckColumnGroup:
    """The kingdom group named by column plus the base supply, distinct.
    Inputs: card_binder, deck_box, column. Output: DeckColumnGroup.
    Side effects: none. Exceptions: as base_supply_uuids."""
    return DeckColumnGroup(
        deck_box,
        column,
        distinct=True,
        always_included=base_supply_uuids(card_binder),
    )


class IsotropicOptionPickDojo(MultiCardOptionSelectionDojo):
    """Options alone in -> the picked option. Subclasses set OUTPUT_PATH
    and PICKS_COLUMN and say how to read the options (Template Method)."""

    OUTPUT_PATH: ClassVar[Path]
    PICKS_COLUMN: ClassVar[str]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs:
            card_binder: card lookup for the Dominion nocab_uuids.
            holdout: card holdout shared by every dojo in a run.
            deck_box: the isotropic metrics' private DeckBox.
            card_embedding_size: width of the encoder's card embeddings.
            path_to_training_data: overrides OUTPUT_PATH.
            name: Trainer-facing name and split-file prefix (see
                DojoConfig.name).
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file or deck box built against another CardBinder.
        Output: none (constructor).
        Side effects: see MultiCardOptionSelectionDojo (may write splits).
        Exceptions: see MultiCardOptionSelectionDojo; ValueError if the
            binder lacks a base supply card.

        Example:
            >>> KingdomOpeningBuyDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            path_to_training_data=path_to_training_data or self.OUTPUT_PATH,
            data_constructor=GroupPickDataConstructor(
                self.option_group(card_binder, deck_box), self.PICKS_COLUMN
            ),
            card_lookup=card_binder,
            holdout=holdout,
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )

    @classmethod
    @abstractmethod
    def option_group(
        cls, card_binder: CardBinder, deck_box: DeckBox
    ) -> DeckColumnGroup:
        """How a row's options are read. Inputs: card_binder, deck_box.
        Output: DeckColumnGroup. Side effects: none. Exceptions: as
        base_supply_uuids."""


class KingdomOpeningBuyDojo(IsotropicOptionPickDojo):
    """Kingdom + base supply -> the eventual winner's opening buy(s)
    (KingdomOpeningBuyPredictionMetric)."""

    OUTPUT_PATH = KingdomOpeningBuyPredictionMetric.DEFAULT_OUTPUT_PATH
    PICKS_COLUMN = "opening_card_uuids"

    @classmethod
    def option_group(
        cls, card_binder: CardBinder, deck_box: DeckBox
    ) -> DeckColumnGroup:
        """See IsotropicOptionPickDojo.option_group."""
        return _kingdom_supply(card_binder, deck_box, "kingdom_uuid")


class KingdomVetoDojo(IsotropicOptionPickDojo):
    """Pre-veto candidate pool -> the vetoed card(s)
    (KingdomVetoPredictionMetric)."""

    OUTPUT_PATH = KingdomVetoPredictionMetric.DEFAULT_OUTPUT_PATH
    PICKS_COLUMN = "vetoed_card_uuids"

    @classmethod
    def option_group(
        cls, card_binder: CardBinder, deck_box: DeckBox
    ) -> DeckColumnGroup:
        """See IsotropicOptionPickDojo.option_group."""
        return DeckColumnGroup(deck_box, "candidate_pool_uuid", distinct=True)


class IsotropicContextPickDojo(MultiGroupOptionSelectionDojo):
    """[options, context] in -> the picked option. Subclasses set
    OUTPUT_PATH and PICKS_COLUMN and say how to read both groups."""

    OUTPUT_PATH: ClassVar[Path]
    PICKS_COLUMN: ClassVar[str]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as IsotropicOptionPickDojo.
        Output: none (constructor).
        Side effects: see MultiGroupOptionSelectionDojo (may write splits).
        Exceptions: see MultiGroupOptionSelectionDojo; ValueError if the
            binder lacks a base supply card.

        Example:
            >>> NextBuyDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            path_to_training_data=path_to_training_data or self.OUTPUT_PATH,
            data_constructor=GroupPickDataConstructor(
                self.option_group(card_binder, deck_box),
                self.PICKS_COLUMN,
                context=DeckColumnGroup(deck_box, "partial_deck_uuid"),
            ),
            card_lookup=card_binder,
            holdout=holdout,
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )

    @classmethod
    @abstractmethod
    def option_group(
        cls, card_binder: CardBinder, deck_box: DeckBox
    ) -> DeckColumnGroup:
        """How a row's options are read; the context is always the
        row's partial deck. Inputs: card_binder, deck_box.
        Output: DeckColumnGroup. Side effects: none."""


class NextBuyDojo(IsotropicContextPickDojo):
    """[kingdom + base supply, partial deck] -> the card(s) that player
    buys this turn (NextBuyPredictionMetric)."""

    OUTPUT_PATH = NextBuyPredictionMetric.DEFAULT_OUTPUT_PATH
    PICKS_COLUMN = "next_buy_card_uuids"

    @classmethod
    def option_group(
        cls, card_binder: CardBinder, deck_box: DeckBox
    ) -> DeckColumnGroup:
        """See IsotropicContextPickDojo.option_group."""
        return _kingdom_supply(card_binder, deck_box, "kingdom_uuid")


class NextTrashedCardDojo(IsotropicContextPickDojo):
    """[the partial deck's distinct cards, the partial deck] -> the
    card(s) trashed this turn (NextTrashedCardMetric). About 7% of
    trashed cards are missing from the rebuilt partial deck (see
    metrics/isotropic/games/TODO.md) and are dropped."""

    OUTPUT_PATH = NextTrashedCardMetric.DEFAULT_OUTPUT_PATH
    PICKS_COLUMN = "next_trashed_card_uuids"

    @classmethod
    def option_group(
        cls, card_binder: CardBinder, deck_box: DeckBox
    ) -> DeckColumnGroup:
        """See IsotropicContextPickDojo.option_group."""
        return DeckColumnGroup(deck_box, "partial_deck_uuid", distinct=True)
