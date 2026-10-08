"""Per-metric wrappers over the isotropic metrics with two card groups
and one value per row: GroupLabelDataConstructor feeding the
multi-group binary or regression cell.

Binary (MultiGroupBinaryClassificationDojo):
- DeckPairWinnerDojo, MidGameDeckPairWinnerDojo: [deck_lo, deck_hi] ->
  did deck_lo win. Symmetric pairs, so TRAIN gets a GroupSwapMod.
- EventualWinDojo: [partial deck, kingdom] -> did that player win.
- OpeningBuyOutcomeDojo: [opening buy, kingdom] -> did that player win.
- WinningDeckMembershipDojo: [[card], kingdom] -> card in the winner's
  final deck.
- KingdomEndingPileDojo: [[card], kingdom] -> card's pile was empty at
  game end.

Regression (MultiGroupRegressionDojo):
- WinningDeckCountDojo: [[card], kingdom] -> copies in the winner's deck.
- DeckCardSetCopyCountDojo: [[card], a deck's distinct card set] -> that
  card's copies in the deck.

The [[card], kingdom] metrics write one row per (kingdom, card), and
DeckCardSetCopyCount one per (deck, card). Those dojos set
SPLIT_GROUP_COLUMN, so all of one kingdom's (or deck's) rows land in one
split instead of leaking across TRAIN and TEST.
"""

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games.kingdom_ending_pile_prediction_metric import (  # noqa: E501
    KingdomEndingPilePredictionMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_deck_pair_winner_metric import (  # noqa: E501
    MidGameDeckPairWinnerMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_win_probability_metric import (  # noqa: E501
    EventualWinProbabilityMetric,
)
from src.data_refinement.metrics.isotropic.games.opening_buy_outcome_metric import (
    OpeningBuyOutcomeMetric,
)
from src.data_refinement.metrics.isotropic.summary.deck_card_set_copy_count_metric import (  # noqa: E501
    DeckCardSetCopyCountMetric,
)
from src.data_refinement.metrics.isotropic.summary.deck_pair_winner_metric import (
    DeckPairWinnerMetric,
)
from src.data_refinement.metrics.isotropic.summary.kingdom_member_label_metrics import (
    WinningDeckCountMetric,
    WinningDeckMembershipMetric,
)
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_group_binary_classification.dojo import (
    MultiGroupBinaryClassificationDojo,
)
from src.dojos.generic.multi_group_binary_classification.group_swap_mod import (
    GroupSwapMod,
)
from src.dojos.generic.multi_group_regression.dojo import MultiGroupRegressionDojo
from src.dojos.isotropic.card_groups import CardColumnGroup, CardGroup, DeckColumnGroup
from src.dojos.isotropic.group_label_data_constructor import GroupLabelDataConstructor
from src.dojos.mods.mod import Mod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec

_KINGDOM_UUID_COLUMN = "kingdom_uuid"
_CARD_UUID_COLUMN = "card_uuid"


class IsotropicGroupBinaryDojo(MultiGroupBinaryClassificationDojo):
    """[group_0, group_1] in -> one 0/1 label out.

    Subclasses set OUTPUT_PATH and LABEL_COLUMN, implement card_groups
    (Template Method), and set SYMMETRIC_PAIR when the label is
    "group 0 beat group 1" (adds a train-only GroupSwapMod).
    """

    OUTPUT_PATH: ClassVar[Path]
    LABEL_COLUMN: ClassVar[str]
    SYMMETRIC_PAIR: ClassVar[bool] = False
    # Split by this column's value (see DojoConfig.split_group_column);
    # None splits row by row.
    SPLIT_GROUP_COLUMN: ClassVar[str | None] = None

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
            rng_seed: split/shuffle seed, also seeding the GroupSwapMod;
                None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file or deck box built against another CardBinder.
        Output: none (constructor).
        Side effects: see MultiGroupBinaryClassificationDojo (may write
            split files).
        Exceptions: see MultiGroupBinaryClassificationDojo.

        Example:
            >>> DeckPairWinnerDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        group_0, group_1 = self.card_groups(deck_box)
        swap: list[Mod] = (
            [GroupSwapMod(rng_seed=rng_seed)] if self.SYMMETRIC_PAIR else []
        )
        super().__init__(
            path_to_training_data=path_to_training_data or self.OUTPUT_PATH,
            data_constructor=GroupLabelDataConstructor(
                group_0, group_1, self.LABEL_COLUMN
            ),
            card_lookup=card_binder,
            holdout=holdout,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(swap),
            deck_box=deck_box,
            config=DojoConfig(
                name=name,
                rng_seed=rng_seed,
                strict_version_check=strict_version_check,
                split_group_column=self.SPLIT_GROUP_COLUMN,
            ),
        )

    @classmethod
    @abstractmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """How a row's two groups are read, in input order.
        Inputs: deck_box. Output: (group_0, group_1).
        Side effects: none. Exceptions: none."""


class IsotropicGroupRegressionDojo(MultiGroupRegressionDojo):
    """[group_0, group_1] in -> one number out. Subclasses set
    OUTPUT_PATH and LABEL_COLUMN and implement card_groups."""

    OUTPUT_PATH: ClassVar[Path]
    LABEL_COLUMN: ClassVar[str]
    # As IsotropicGroupBinaryDojo.SPLIT_GROUP_COLUMN.
    SPLIT_GROUP_COLUMN: ClassVar[str | None] = None

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
        objective: RegressionObjective | None = None,
    ) -> None:
        """
        Inputs: as IsotropicGroupBinaryDojo.
        Output: none (constructor).
        Side effects: see MultiGroupRegressionDojo (may write splits).
        Exceptions: see MultiGroupRegressionDojo.

        Example:
            >>> WinningDeckCountDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        group_0, group_1 = self.card_groups(deck_box)
        super().__init__(
            path_to_training_data=path_to_training_data or self.OUTPUT_PATH,
            data_constructor=GroupLabelDataConstructor(
                group_0, group_1, self.LABEL_COLUMN
            ),
            card_lookup=card_binder,
            holdout=holdout,
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            objective=objective,
            config=DojoConfig(
                name=name,
                rng_seed=rng_seed,
                strict_version_check=strict_version_check,
                split_group_column=self.SPLIT_GROUP_COLUMN,
            ),
        )

    @classmethod
    @abstractmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """How a row's two groups are read, in input order.
        Inputs: deck_box. Output: (group_0, group_1).
        Side effects: none. Exceptions: none."""


def _deck_pair(deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
    """(deck_uuid_lo's deck, deck_uuid_hi's deck). Inputs: deck_box.
    Output: the two groups. Side effects: none. Exceptions: none."""
    return (
        DeckColumnGroup(deck_box, "deck_uuid_lo"),
        DeckColumnGroup(deck_box, "deck_uuid_hi"),
    )


def _card_in_kingdom(deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
    """([card_uuid's card], kingdom_uuid's kingdom). Inputs: deck_box.
    Output: the two groups. Side effects: none. Exceptions: none."""
    return (
        CardColumnGroup(_CARD_UUID_COLUMN),
        DeckColumnGroup(deck_box, _KINGDOM_UUID_COLUMN),
    )


class DeckPairWinnerDojo(IsotropicGroupBinaryDojo):
    """Both final decks of a 2-player game -> did the lower-uuid deck win
    (DeckPairWinnerMetric)."""

    OUTPUT_PATH = DeckPairWinnerMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "lo_won"
    SYMMETRIC_PAIR = True

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupBinaryDojo.card_groups."""
        return _deck_pair(deck_box)


class MidGameDeckPairWinnerDojo(IsotropicGroupBinaryDojo):
    """Both players' partial decks at one turn -> did the lower-uuid deck
    win (MidGameDeckPairWinnerMetric)."""

    OUTPUT_PATH = MidGameDeckPairWinnerMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "lo_wins"
    SYMMETRIC_PAIR = True

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupBinaryDojo.card_groups."""
        return _deck_pair(deck_box)


class EventualWinDojo(IsotropicGroupBinaryDojo):
    """[partial deck, kingdom] -> did that player eventually win
    (EventualWinProbabilityMetric)."""

    OUTPUT_PATH = EventualWinProbabilityMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "player_wins"

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupBinaryDojo.card_groups."""
        return (
            DeckColumnGroup(deck_box, "partial_deck_uuid"),
            DeckColumnGroup(deck_box, _KINGDOM_UUID_COLUMN),
        )


class OpeningBuyOutcomeDojo(IsotropicGroupBinaryDojo):
    """[a player's opening buy, kingdom] -> did that player win
    (OpeningBuyOutcomeMetric)."""

    OUTPUT_PATH = OpeningBuyOutcomeMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "won"

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupBinaryDojo.card_groups."""
        return (
            DeckColumnGroup(deck_box, "opening_group_uuid"),
            DeckColumnGroup(deck_box, _KINGDOM_UUID_COLUMN),
        )


class WinningDeckMembershipDojo(IsotropicGroupBinaryDojo):
    """[[kingdom card], kingdom] -> is it in the winner's final deck
    (WinningDeckMembershipMetric)."""

    OUTPUT_PATH = WinningDeckMembershipMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = WinningDeckMembershipMetric.LABEL_COLUMN
    SPLIT_GROUP_COLUMN = _KINGDOM_UUID_COLUMN

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupBinaryDojo.card_groups."""
        return _card_in_kingdom(deck_box)


class KingdomEndingPileDojo(IsotropicGroupBinaryDojo):
    """[[kingdom card], kingdom] -> was its pile empty at game end
    (KingdomEndingPilePredictionMetric). About 5% positive."""

    OUTPUT_PATH = KingdomEndingPilePredictionMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "exhausted"
    SPLIT_GROUP_COLUMN = _KINGDOM_UUID_COLUMN

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupBinaryDojo.card_groups."""
        return _card_in_kingdom(deck_box)


class WinningDeckCountDojo(IsotropicGroupRegressionDojo):
    """[[kingdom card], kingdom] -> its copies in the winner's final deck
    (WinningDeckCountMetric)."""

    OUTPUT_PATH = WinningDeckCountMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = WinningDeckCountMetric.LABEL_COLUMN
    SPLIT_GROUP_COLUMN = _KINGDOM_UUID_COLUMN

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupRegressionDojo.card_groups."""
        return _card_in_kingdom(deck_box)


class DeckCardSetCopyCountDojo(IsotropicGroupRegressionDojo):
    """[[card], a final deck's distinct card set] -> that card's copies
    in the deck (DeckCardSetCopyCountMetric)."""

    OUTPUT_PATH = DeckCardSetCopyCountMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "count"
    SPLIT_GROUP_COLUMN = "deck_set_uuid"

    @classmethod
    def card_groups(cls, deck_box: DeckBox) -> tuple[CardGroup, CardGroup]:
        """See IsotropicGroupRegressionDojo.card_groups."""
        return (
            CardColumnGroup(_CARD_UUID_COLUMN),
            DeckColumnGroup(deck_box, "deck_set_uuid"),
        )
