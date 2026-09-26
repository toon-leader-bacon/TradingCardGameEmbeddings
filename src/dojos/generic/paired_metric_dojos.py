"""Base classes for the per-metric dojo wrappers, one per recurring
(metric family shape, generic dojo cell) pairing.

A per-metric wrapper (e.g. sts_gg's CardRelicCountDojo) adds no
behavior of its own: it points one generic dojo cell at its paired
metric class's parquet output and label column. Every wrapper of the
same shape built its parent identically, so each shape's constructor
lives here once (Template Method) and a wrapper only sets METRIC:

    class CardRelicCountDojo(CardAverageMetricDojo):
        \"\"\"Card -> predicted average relicCount.\"\"\"

        METRIC = CardRelicCountMetric

A wrapper never instantiates METRIC (its constructor needs a
card_binder/deck_box a dojo has no reason to fabricate); it only reads
METRIC's ClassVars, so they are never duplicated as literals.
"""

from pathlib import Path
from typing import ClassVar, Protocol, Sequence

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors import (
    CardAverageDataConstructor,
    DeckLabelDataConstructor,
    MaskedFieldDataConstructor,
)
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec


class LabelColumnMetric(Protocol):
    """A metric class whose parquet output has one label column."""

    DEFAULT_OUTPUT_PATH: Path
    LABEL_COLUMN: str


class MaskedFieldMetricClass(Protocol):
    """A MaskedFieldMetric subclass (see
    src/data_refinement/metrics/generic/masked_field_metric.py)."""

    DEFAULT_OUTPUT_PATH: Path
    LABEL_VALUES: tuple[str, ...]
    MASKED_FIELD: list[str]


class CardAverageMetricDojo(SingleCardRegressionDojo):
    """Single card in -> METRIC.LABEL_COLUMN (a per-card average) out.

    Subclasses set METRIC to a metric class with DEFAULT_OUTPUT_PATH and
    LABEL_COLUMN (CardAverageMetric and its 17lands counterparts).
    """

    METRIC: ClassVar[LabelColumnMetric]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs:
            card_binder: card lookup for the metric's nocab_uuids.
            holdout: card holdout shared by every dojo in a run.
            card_embedding_size: width of the encoder's card embeddings.
            path_to_training_data: overrides METRIC.DEFAULT_OUTPUT_PATH.
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file built against a different CardBinder version.
        Output: none (constructor).
        Side effects: see SingleCardRegressionDojo (may write split files).
        Exceptions: see SingleCardRegressionDojo.

        Example:
            >>> CardRelicCountDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(self.METRIC.LABEL_COLUMN),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class DeckLabelMetricDojo(MultiCardRegressionDojo):
    """Whole deck in -> METRIC.LABEL_COLUMN (a per-deck number) out.

    Subclasses set METRIC to a DeckLabelMetric-shaped class with
    DEFAULT_OUTPUT_PATH and LABEL_COLUMN; each row's deck is looked up
    in the given DeckBox.
    """

    METRIC: ClassVar[LabelColumnMetric]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as CardAverageMetricDojo, plus deck_box (DeckBox), the
            store the metric's deck nocab_uuids point into.
        Output: none (constructor).
        Side effects: see MultiCardRegressionDojo (may write split files).
        Exceptions: see MultiCardRegressionDojo.

        Example:
            >>> DeckRelicCountDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                deck_box, self.METRIC.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class MaskedFieldMetricDojo(SingleCardFixedClassificationDojo):
    """Single card with METRIC's field masked in -> that field's value out.

    Subclasses set METRIC to a MaskedFieldMetric subclass. The mask is
    applied with train_only=False: the field must stay masked on
    test/validation input too, or the model could read the answer off
    raw_content. EXTRA_MASKED_KEYS names further top-level keys that
    leak the answer and must be masked alongside it.

    MaskTargetKeyMod masks one top-level key, so this masks
    MASKED_FIELD[-1]; every MaskedFieldMetric today has a single-element
    MASKED_FIELD. A nested one would need MaskTargetKeyMod to walk a
    path first.
    """

    METRIC: ClassVar[MaskedFieldMetricClass]
    EXTRA_MASKED_KEYS: ClassVar[Sequence[str]] = ()

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as CardAverageMetricDojo.
        Output: none (constructor).
        Side effects: see SingleCardFixedClassificationDojo (may write
            split files).
        Exceptions: see SingleCardFixedClassificationDojo.

        Example:
            >>> ColorMaskDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        masked_keys = [self.METRIC.MASKED_FIELD[-1], *self.EXTRA_MASKED_KEYS]
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor("label"),
            label_values=self.METRIC.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [MaskTargetKeyMod(key=key, train_only=False) for key in masked_keys]
            ),
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
