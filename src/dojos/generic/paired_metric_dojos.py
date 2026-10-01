"""Base classes for the per-metric dojo wrappers, one per recurring
(metric family shape, generic dojo cell) pairing.

A per-metric wrapper (e.g. sts_gg's CardRelicCountDojo) adds no
behavior of its own: it points one generic dojo cell at its paired
metric class's parquet output and label column. Every wrapper of the
same shape built its parent identically, so each shape's constructor
lives here once (Template Method) and a wrapper only sets METRIC (plus,
for the three masked-field bases, any other keys that leak the answer):

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
from src.data_refinement.metrics.generic.masked_field_metric import MaskedFieldMetric
from src.data_refinement.metrics.generic.masked_field_multi_label_metric import (
    MaskedFieldMultiLabelMetric,
)
from src.data_refinement.metrics.generic.masked_field_regression_metric import (
    MaskedFieldRegressionMetric,
)
from src.dojos.generic.data_constructors import (
    CardAverageDataConstructor,
    DeckLabelDataConstructor,
    MaskedFieldDataConstructor,
    MaskedFieldMultiLabelDataConstructor,
    MaskedFieldRegressionDataConstructor,
)
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.generic.single_card_fixed_classification.loss_spec import (
    MASKED_VECTOR_REGRESSION_LOSS_SPEC,
)
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.card_factory import FieldPath, MissingPathPolicy
from src.schema.holdout import HoldoutSpec


class LabelColumnMetric(Protocol):
    """A metric class whose parquet output has one label column."""

    DEFAULT_OUTPUT_PATH: Path
    LABEL_COLUMN: str


def _field_masking_pipeline(
    masked_field: Sequence[str],
    extra_keys: Sequence[str],
    extra_paths: Sequence[FieldPath],
) -> ModPipeline:
    """The masking mods every masked-field wrapper applies on every split
    (train_only=False: a field left visible on test/validation input
    would let the model read the answer off raw_content).

    Inputs:
        masked_field: the metric's MASKED_FIELD; its last key is masked
            (every masked-field metric today has a one-element path).
        extra_keys: further top-level keys that leak the answer; masked
            on every card, added as "[MASK]" where absent so every
            card's input has the same keys.
        extra_paths: nested FieldPaths that leak the answer but exist on
            only some cards (e.g. ("attacks", 1, "cost")); masked where
            present and skipped where not (MissingPathPolicy.PASS),
            since a missing list slot cannot be added.
    Output: ModPipeline of MaskTargetKeyMods, the field's own first.
    Side effects: none. Exceptions: none.

    Example:
        >>> _field_masking_pipeline(["hp"], (), ()).mods[0].key
        'hp'
    """
    keys = [masked_field[-1], *extra_keys]
    key_mods = [MaskTargetKeyMod(key=key, train_only=False) for key in keys]
    path_mods = [
        MaskTargetKeyMod(key=path, train_only=False, missing=MissingPathPolicy.PASS)
        for path in extra_paths
    ]
    return ModPipeline([*key_mods, *path_mods])


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
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs:
            card_binder: card lookup for the metric's nocab_uuids.
            holdout: card holdout shared by every dojo in a run.
            card_embedding_size: width of the encoder's card embeddings.
            path_to_training_data: overrides METRIC.DEFAULT_OUTPUT_PATH.
            name: this dojo's Trainer-facing name and split-file prefix;
                None falls back to path_to_training_data.stem (see
                DojoConfig.name). Set explicitly whenever more than one
                dojo of this class is built from metric files that share
                a bare filename, to avoid a split-file collision.
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
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
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
        name: str | None = None,
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
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class MaskedFieldMetricDojo(SingleCardFixedClassificationDojo):
    """Single card with METRIC's field masked in -> that field's value out.

    Subclasses set METRIC to a MaskedFieldMetric subclass. The mask is
    applied with train_only=False: the field must stay masked on
    test/validation input too, or the model could read the answer off
    raw_content. EXTRA_MASKED_KEYS names further top-level keys that
    leak the answer and must be masked alongside it; EXTRA_MASKED_PATHS
    names nested slots that leak it where present (see
    _field_masking_pipeline).

    This masks MASKED_FIELD[-1] as a top-level key; every masked-field
    metric today has a single-element MASKED_FIELD.

    Each masked-field base types METRIC as its own metric family, so a
    wrapper cannot pair (say) a multi-label metric with a single-label
    cell, whose labels would then be read as the wrong shape.
    """

    METRIC: ClassVar[type[MaskedFieldMetric]]
    EXTRA_MASKED_KEYS: ClassVar[Sequence[str]] = ()
    EXTRA_MASKED_PATHS: ClassVar[Sequence[FieldPath]] = ()

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
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
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor("label"),
            label_values=self.METRIC.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=_field_masking_pipeline(
                self.METRIC.MASKED_FIELD,
                self.EXTRA_MASKED_KEYS,
                self.EXTRA_MASKED_PATHS,
            ),
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class MaskedFieldRegressionMetricDojo(SingleCardRegressionDojo):
    """Single card with METRIC's field masked in -> that field's number out.

    Subclasses set METRIC to a MaskedFieldRegressionMetric subclass, and
    optionally EXTRA_MASKED_KEYS / EXTRA_MASKED_PATHS exactly as for
    MaskedFieldMetricDojo.
    """

    METRIC: ClassVar[type[MaskedFieldRegressionMetric]]
    EXTRA_MASKED_KEYS: ClassVar[Sequence[str]] = ()
    EXTRA_MASKED_PATHS: ClassVar[Sequence[FieldPath]] = ()

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as CardAverageMetricDojo.
        Output: none (constructor).
        Side effects: see SingleCardRegressionDojo (may write split files).
        Exceptions: see SingleCardRegressionDojo.

        Example:
            >>> HpRegressionDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldRegressionDataConstructor("label"),
            card_embedding_size=card_embedding_size,
            mod_pipeline=_field_masking_pipeline(
                self.METRIC.MASKED_FIELD,
                self.EXTRA_MASKED_KEYS,
                self.EXTRA_MASKED_PATHS,
            ),
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class MaskedFieldMultiLabelMetricDojo(SingleCardFixedClassificationDojo):
    """Single card with METRIC's field masked in -> one independent
    probability per METRIC.LABEL_VALUES member out (e.g. MTG colors).

    Subclasses set METRIC to a MaskedFieldMultiLabelMetric subclass, and
    optionally EXTRA_MASKED_KEYS / EXTRA_MASKED_PATHS exactly as for
    MaskedFieldMetricDojo. Scored with MASKED_VECTOR_REGRESSION_LOSS_SPEC
    (a sigmoid per member, MSE against 0/1; baseline: each member's
    TRAIN rate): the members do not compete for one probability mass
    the way softmax classes do, so a two-color card is two 1.0s.
    """

    METRIC: ClassVar[type[MaskedFieldMultiLabelMetric]]
    EXTRA_MASKED_KEYS: ClassVar[Sequence[str]] = ()
    EXTRA_MASKED_PATHS: ClassVar[Sequence[FieldPath]] = ()

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
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
            >>> ColorsMaskDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldMultiLabelDataConstructor(
                self.METRIC.LABEL_VALUES
            ),
            label_values=self.METRIC.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=_field_masking_pipeline(
                self.METRIC.MASKED_FIELD,
                self.EXTRA_MASKED_KEYS,
                self.EXTRA_MASKED_PATHS,
            ),
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
            loss_spec=MASKED_VECTOR_REGRESSION_LOSS_SPEC,
        )
