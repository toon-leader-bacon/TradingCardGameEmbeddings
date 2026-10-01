"""Per-metric wrappers over the isotropic metrics whose input is one card
group from the metrics-private DeckBox (data/metrics/isotropic/deck_box.db)
and whose label is one value per row - the DeckLabelDataConstructor shape.

- FullDeckWinPredictionDojo: final deck -> won (binary).
- KingdomGameLengthDojo: kingdom -> winner's turn count (regression).
- NextTurnActionCountDojo: mid-game partial deck -> Action cards played
  next turn (regression).
- KingdomEndingTypeDojo: kingdom -> province / colony / multi_pile
  (fixed classification).

DECK COLUMN: DeckLabelDataConstructor reads a fixed "deck_uuid" column;
the kingdom and partial-deck metrics name theirs kingdom_uuid and
partial_deck_uuid, so those wrappers wrap it in a
RenamedColumnDataConstructor. Label column names are literals in the
metrics' parquet schemas (no LABEL_COLUMN ClassVar), so they are named
once here as module constants.

DECK BOX: every wrapper passes deck_box to its cell as well as to its
constructor, so GenericDojo's version check can confirm the box and the
metric were built from the same CardBinder.
"""

from pathlib import Path
from typing import Callable, ClassVar

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.games import kingdom_ending_type_metric
from src.data_refinement.metrics.isotropic.games.kingdom_ending_type_metric import (
    KingdomEndingTypeMetric,
)
from src.data_refinement.metrics.isotropic.games.mid_game_next_turn_action_count_metric import (
    NextTurnActionCountMetric,
)
from src.data_refinement.metrics.isotropic.summary.full_deck_win_prediction_metric import (
    FullDeckWinPredictionMetric,
)
from src.data_refinement.metrics.isotropic.summary.kingdom_game_length_metric import (
    KingdomGameLengthMetric,
)
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.isotropic.renamed_column_data_constructor import (
    RenamedColumnDataConstructor,
)
from src.schema.holdout import HoldoutSpec
from src.schema.type_hints import Label

_DECK_UUID_COLUMN = "deck_uuid"
_KINGDOM_UUID_COLUMN = "kingdom_uuid"
_PARTIAL_DECK_UUID_COLUMN = "partial_deck_uuid"

_WON_COLUMN = "won"
_WINNER_TURNS_COLUMN = "winner_turns"
_NEXT_TURN_ACTION_COUNT_COLUMN = "next_turn_action_count"
_ENDING_TYPE_COLUMN = "ending_type"


def build_deck_label_constructor(
    deck_box: DeckBox,
    deck_column: str,
    label_column: str,
    label_caster: Callable[[object], Label] | None = None,
) -> DataConstructor:
    """A DeckLabelDataConstructor for a metric whose group column is
    deck_column, renamed to "deck_uuid" first when it differs.

    Inputs:
        deck_box: the DeckBox the metric's group uuids point into.
        deck_column: the metric's group uuid column name.
        label_column: the metric's label column name.
        label_caster: converts a raw label cell; None keeps
            DeckLabelDataConstructor's float default (regression and
            binary cells), str suits a fixed-classification cell.
    Output: DataConstructor - a plain DeckLabelDataConstructor when
        deck_column is already "deck_uuid", else one wrapped in a
        RenamedColumnDataConstructor.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> build_deck_label_constructor(box, "kingdom_uuid", "winner_turns")
    """
    result: DataConstructor = (
        DeckLabelDataConstructor(deck_box, label_column)
        if label_caster is None
        else DeckLabelDataConstructor(deck_box, label_column, label_caster)
    )
    if deck_column != _DECK_UUID_COLUMN:
        result = RenamedColumnDataConstructor(result, {deck_column: _DECK_UUID_COLUMN})
    return result


class IsotropicDeckRegressionDojo(MultiCardRegressionDojo):
    """Card group in -> one number out, for an isotropic metric keyed by
    DECK_COLUMN with its label in LABEL_COLUMN.

    Subclasses set OUTPUT_PATH (read off the metric class), DECK_COLUMN
    and LABEL_COLUMN.
    """

    OUTPUT_PATH: ClassVar[Path]
    DECK_COLUMN: ClassVar[str]
    LABEL_COLUMN: ClassVar[str]

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
            name: Trainer-facing name and split-file prefix; None falls
                back to the training file's stem (see DojoConfig.name).
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file or deck box built against another CardBinder.
        Output: none (constructor).
        Side effects: see MultiCardRegressionDojo (may write split files).
        Exceptions: see MultiCardRegressionDojo.

        Example:
            >>> KingdomGameLengthDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data or self.OUTPUT_PATH,
            data_constructor=build_deck_label_constructor(
                deck_box, self.DECK_COLUMN, self.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class KingdomGameLengthDojo(IsotropicDeckRegressionDojo):
    """Kingdom (the supply cards) -> the winner's turn count
    (KingdomGameLengthMetric). Labels run 0-323 with a ~2% tail under 5
    turns (early resignations); MSE is sensitive to that tail."""

    OUTPUT_PATH = KingdomGameLengthMetric.DEFAULT_OUTPUT_PATH
    DECK_COLUMN = _KINGDOM_UUID_COLUMN
    LABEL_COLUMN = _WINNER_TURNS_COLUMN


class NextTurnActionCountDojo(IsotropicDeckRegressionDojo):
    """Mid-game partial deck -> number of Action cards that player plays
    on their next turn (NextTurnActionCountMetric)."""

    OUTPUT_PATH = NextTurnActionCountMetric.DEFAULT_OUTPUT_PATH
    DECK_COLUMN = _PARTIAL_DECK_UUID_COLUMN
    LABEL_COLUMN = _NEXT_TURN_ACTION_COUNT_COLUMN


class FullDeckWinPredictionDojo(MultiCardBinaryClassificationDojo):
    """One player's final deck -> finished first (FullDeckWinPredictionMetric).

    The final deck holds its Victory cards, so part of this is learning
    which cards score; that is intended signal, not a leak."""

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
        Inputs: as IsotropicDeckRegressionDojo.
        Output: none (constructor).
        Side effects: see MultiCardBinaryClassificationDojo (may write
            split files).
        Exceptions: see MultiCardBinaryClassificationDojo.

        Example:
            >>> FullDeckWinPredictionDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or FullDeckWinPredictionMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=build_deck_label_constructor(
                deck_box, _DECK_UUID_COLUMN, _WON_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class KingdomEndingTypeDojo(MultiCardFixedClassificationDojo):
    """Kingdom -> how the game ended: "province", "colony" or "multi_pile"
    (KingdomEndingTypeMetric). label_values is that metric module's own
    LABEL_VALUES, read by reference."""

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
        Inputs: as IsotropicDeckRegressionDojo.
        Output: none (constructor).
        Side effects: see MultiCardFixedClassificationDojo (may write
            split files).
        Exceptions: see MultiCardFixedClassificationDojo.

        Example:
            >>> KingdomEndingTypeDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or KingdomEndingTypeMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=build_deck_label_constructor(
                deck_box, _KINGDOM_UUID_COLUMN, _ENDING_TYPE_COLUMN, label_caster=str
            ),
            label_values=kingdom_ending_type_metric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
