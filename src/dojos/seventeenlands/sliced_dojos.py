"""Slice-aware plumbing every 17lands dojo shares.

seventeenlands_training_path() is the one place the slice-or-override
rule lives: an explicit path wins (tests, one-off files), else the
metric's slice file is built (or reused) and its path returned.

Four intermediate bases cover the wrappers that only name their metric,
the slice-aware twins of the generic CardAverageMetricDojo and
../generic/paired_metric_dojos.py's deck bases (which other sources
share, read METRIC.DEFAULT_OUTPUT_PATH, and stay unchanged):

- SeventeenLandsCardLabelDojo: card -> METRIC.LABEL_COLUMN, for every
  per-card count table.
- SeventeenLandsDeckRegressionDojo: deck -> METRIC.LABEL_COLUMN, for a
  per-deck count table or a per-game row stream with a numeric label.
- SeventeenLandsDeckBinaryLabelDojo: deck -> a boolean label.
- SeventeenLandsDeckFixedLabelDojo: deck -> one of METRIC.LABEL_VALUES.

Hand-written wrappers (option selection, multi-group, curve cells) call
seventeenlands_training_path() directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.slice_file import (
    SeventeenLandsSliceFile,
)
from src.data_refinement.metrics.seventeenlands.sliced_metric import (
    CountTableMetric,
    RowStreamMetric,
    SlicedMetricClass,
)
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.generic.data_constructors import (
    CardAverageDataConstructor,
    DeckLabelDataConstructor,
)
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec

ALL_DATA = SeventeenLandsSlice()


class FixedLabelRowStreamMetric(RowStreamMetric, Protocol):
    """A row stream whose str label is one of LABEL_VALUES (OTHER
    included where the metric has one)."""

    LABEL_VALUES: ClassVar[tuple[str, ...]]


def seventeenlands_training_path(
    metric: SlicedMetricClass,
    data_slice: SeventeenLandsSlice,
    path_to_training_data: Path | None,
) -> Path:
    """The parquet file a 17lands dojo trains on.

    Inputs:
        metric: the dojo's metric class.
        data_slice: the sets and formats to train on.
        path_to_training_data: an explicit file; when given, data_slice
            is ignored and nothing is built.
    Output: path_to_training_data, else
        SeventeenLandsSliceFile(metric).build(data_slice).
    Side effects: may build the slice file (SeventeenLandsSliceFile.build).
    Exceptions: as SeventeenLandsSliceFile.build.

    Example:
        >>> seventeenlands_training_path(DrawnWinRateMetric, ALL_DATA, None)
        PosixPath('data/metrics/seventeenlands/game_data/slices/drawn_win_rate.all.parquet')
    """
    if path_to_training_data is not None:
        return path_to_training_data
    return SeventeenLandsSliceFile(metric).build(data_slice)


class SeventeenLandsCardLabelDojo(SingleCardRegressionDojo):
    """Single card in -> METRIC.LABEL_COLUMN out, from a per-card count
    table's slice file.

    Subclasses set METRIC.
    """

    METRIC: ClassVar[type[CountTableMetric]]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        data_slice: SeventeenLandsSlice = ALL_DATA,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
        objective: RegressionObjective | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: card lookup for the metric's nocab_uuids.
            holdout: card holdout shared by every dojo in a run.
            card_embedding_size: width of the encoder's card embeddings.
            data_slice: the sets and formats to train on (default all).
            path_to_training_data: an explicit file; overrides
                data_slice.
            name: Trainer-facing name and split prefix; None falls back
                to the file stem (<OUTPUT_STEM>.<slice name>).
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a binder
                version mismatch.
            objective: the loss and baseline to train with; None is plain
                MSE with the mean predictor's baseline (RegressionObjective.mse()).
        Output: none (constructor).
        Side effects: may build the slice file; see
            SingleCardRegressionDojo (may write split files).
        Exceptions: as seventeenlands_training_path and
            SingleCardRegressionDojo.

        Example:
            >>> DrawnWinRateDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=seventeenlands_training_path(
                self.METRIC, data_slice, path_to_training_data
            ),
            data_constructor=CardAverageDataConstructor(self.METRIC.LABEL_COLUMN),
            card_embedding_size=card_embedding_size,
            objective=objective,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class SeventeenLandsDeckRegressionDojo(MultiCardRegressionDojo):
    """Whole deck in -> METRIC.LABEL_COLUMN out, each row's deck_uuid
    looked up in the family deck box.

    Subclasses set METRIC.
    """

    METRIC: ClassVar[SlicedMetricClass]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        data_slice: SeventeenLandsSlice = ALL_DATA,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
        objective: RegressionObjective | None = None,
    ) -> None:
        """
        Inputs: as SeventeenLandsCardLabelDojo, plus deck_box (the
            family DeckBox the metric's deck_uuids point into).
        Output: none (constructor).
        Side effects: may build the slice file; see
            MultiCardRegressionDojo (may write split files).
        Exceptions: as seventeenlands_training_path and
            MultiCardRegressionDojo.

        Example:
            >>> DeckGameLengthPredictionDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=seventeenlands_training_path(
                self.METRIC, data_slice, path_to_training_data
            ),
            data_constructor=DeckLabelDataConstructor(
                deck_box, self.METRIC.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            objective=objective,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class SeventeenLandsDeckBinaryLabelDojo(MultiCardBinaryClassificationDojo):
    """Whole deck in -> a boolean METRIC.LABEL_COLUMN out (the default
    label_caster=float maps True/False to 1.0/0.0), each row's deck_uuid
    looked up in deck_box. The slice-aware twin of
    ../generic/paired_metric_dojos.py's DeckBinaryLabelMetricDojo.

    Subclasses set METRIC.
    """

    METRIC: ClassVar[SlicedMetricClass]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        data_slice: SeventeenLandsSlice = ALL_DATA,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as SeventeenLandsDeckRegressionDojo.
        Output: none (constructor).
        Side effects: may build the slice file; see
            MultiCardBinaryClassificationDojo (may write split files).
        Exceptions: as seventeenlands_training_path and
            MultiCardBinaryClassificationDojo.

        Example:
            >>> DeckWinPredictionDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=seventeenlands_training_path(
                self.METRIC, data_slice, path_to_training_data
            ),
            data_constructor=DeckLabelDataConstructor(
                deck_box, self.METRIC.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class SeventeenLandsDeckFixedLabelDojo(MultiCardFixedClassificationDojo):
    """Whole deck in -> one of METRIC.LABEL_VALUES out (a str label),
    each row's deck_uuid looked up in deck_box. The slice-aware twin of
    ../generic/paired_metric_dojos.py's DeckFixedLabelMetricDojo.

    Subclasses set METRIC to a metric with LABEL_VALUES.
    """

    METRIC: ClassVar[type[FixedLabelRowStreamMetric]]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        data_slice: SeventeenLandsSlice = ALL_DATA,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as SeventeenLandsDeckRegressionDojo.
        Output: none (constructor).
        Side effects: may build the slice file; see
            MultiCardFixedClassificationDojo (may write split files).
        Exceptions: as seventeenlands_training_path and
            MultiCardFixedClassificationDojo.

        Example:
            >>> DeckRankTierPredictionDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=seventeenlands_training_path(
                self.METRIC, data_slice, path_to_training_data
            ),
            data_constructor=DeckLabelDataConstructor(
                deck_box, self.METRIC.LABEL_COLUMN, label_caster=str
            ),
            label_values=self.METRIC.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
