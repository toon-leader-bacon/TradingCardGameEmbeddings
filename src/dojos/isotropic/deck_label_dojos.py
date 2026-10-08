"""Per-metric wrappers over the isotropic metrics whose input is one card
group from the metrics-private DeckBox (data/metrics/isotropic/deck_box.db)
and whose label is one value per row - the DeckLabelDataConstructor shape.

- FullDeckWinPredictionDojo: final deck -> won (binary); a thin
  DeckBinaryLabelMetricDojo subclass (see
  src/dojos/generic/paired_metric_dojos.py) naming only METRIC, since
  its deck column is already "deck_uuid" and its metric declares
  LABEL_COLUMN.
- KingdomGameLengthDojo: kingdom -> winner's turn count (regression).
- NextTurnActionCountDojo: mid-game partial deck -> Action cards played
  next turn (regression).
- KingdomEndingTypeDojo: kingdom -> province / colony / multi_pile
  (fixed classification).

DECK COLUMN: DeckLabelDataConstructor reads a fixed "deck_uuid" column;
the kingdom and partial-deck metrics name theirs kingdom_uuid and
partial_deck_uuid, so those wrappers wrap it in a
RenamedColumnDataConstructor. Their label column names are literals in
the metrics' parquet schemas (no LABEL_COLUMN ClassVar), so they are
named once here as module constants.

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
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.generic.paired_metric_dojos import DeckBinaryLabelMetricDojo
from src.dojos.generic.renamed_column_data_constructor import (
    RenamedColumnDataConstructor,
)
from src.schema.holdout import HoldoutSpec
from src.schema.type_hints import Label

_DECK_UUID_COLUMN = "deck_uuid"
_KINGDOM_UUID_COLUMN = "kingdom_uuid"
_PARTIAL_DECK_UUID_COLUMN = "partial_deck_uuid"

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


def label_capped_at(cap: float) -> Callable[[object], float]:
    """A label_caster that reads a numeric cell as a float clipped to at
    most cap.

    Inputs: cap (float). Output: Callable[[object], float].
    Side effects: none. Exceptions: the returned caster raises
        TypeError/ValueError on a non-numeric cell.

    Example:
        >>> label_capped_at(30.0)(112)
        30.0
    """

    def caster(raw_label: object) -> float:
        return min(float(raw_label), cap)  # type: ignore[arg-type]  # numeric cells

    return caster


class IsotropicDeckRegressionDojo(MultiCardRegressionDojo):
    """Card group in -> one number out, for an isotropic metric keyed by
    DECK_COLUMN with its label in LABEL_COLUMN.

    Subclasses set OUTPUT_PATH (read off the metric class), DECK_COLUMN,
    LABEL_COLUMN and LABEL_CAP (labels above it are clipped to it, so a
    handful of extreme labels cannot dominate a z-scored MSE).
    """

    OUTPUT_PATH: ClassVar[Path]
    DECK_COLUMN: ClassVar[str]
    LABEL_COLUMN: ClassVar[str]
    LABEL_CAP: ClassVar[float]

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
            objective: the loss and baseline to train with; None is plain
                MSE with the mean predictor's baseline (RegressionObjective.mse()).
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
                deck_box,
                self.DECK_COLUMN,
                self.LABEL_COLUMN,
                label_caster=label_capped_at(self.LABEL_CAP),
            ),
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            objective=objective,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class KingdomGameLengthDojo(IsotropicDeckRegressionDojo):
    """Kingdom (the supply cards) -> the winner's turn count
    (KingdomGameLengthMetric). The metric skips solo and resigned games;
    one 2013 day of what is left runs 10-51 turns, so the cap of 50 only
    guards against a stray stalled game (an older file reached 323)."""

    OUTPUT_PATH = KingdomGameLengthMetric.DEFAULT_OUTPUT_PATH
    DECK_COLUMN = _KINGDOM_UUID_COLUMN
    LABEL_COLUMN = _WINNER_TURNS_COLUMN
    LABEL_CAP = 50.0


class NextTurnActionCountDojo(IsotropicDeckRegressionDojo):
    """Mid-game partial deck -> number of Action cards that player plays
    on their next turn (NextTurnActionCountMetric). Labels reach 112
    (village/King's Court chains) with p99.9 = 27, so they are clipped at
    30 (0.07% of rows)."""

    OUTPUT_PATH = NextTurnActionCountMetric.DEFAULT_OUTPUT_PATH
    DECK_COLUMN = _PARTIAL_DECK_UUID_COLUMN
    LABEL_COLUMN = _NEXT_TURN_ACTION_COUNT_COLUMN
    LABEL_CAP = 30.0


class FullDeckWinPredictionDojo(DeckBinaryLabelMetricDojo):
    """One player's final deck -> finished first (FullDeckWinPredictionMetric).

    The final deck holds its Victory cards, so part of this is learning
    which cards score; that is intended signal, not a leak."""

    METRIC = FullDeckWinPredictionMetric


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
