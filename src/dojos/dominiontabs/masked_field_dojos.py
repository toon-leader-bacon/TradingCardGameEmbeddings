"""Per-metric wrappers over dominiontabs' two classification metrics
(src/data_refinement/metrics/dominiontabs/{type,set}_mask_metric.py).

TypeMaskDojo is a MaskedFieldMetricDojo (../generic/paired_metric_dojos.py).
SetMaskDojo is not: its metric is not a MaskedFieldMetric and there is
nothing to mask (see its own docstring).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.dominiontabs.set_mask_metric import SetMaskMetric
from src.data_refinement.metrics.dominiontabs.type_mask_metric import TypeMaskMetric
from src.dojos.generic.paired_metric_dojos import MaskedFieldMetricDojo
from src.dojos.generic.data_constructors import MaskedFieldDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.schema.holdout import HoldoutSpec


class TypeMaskDojo(MaskedFieldMetricDojo):
    """Card, primary type masked -> predicted primary type
    (TypeMaskMetric). label_values=TypeMaskMetric.LABEL_VALUES
    unchanged - OTHER_LABEL is already baked into that tuple."""

    METRIC = TypeMaskMetric


class SetMaskDojo(SingleCardFixedClassificationDojo):
    """Card -> predicted expansion (SetMaskMetric).
    label_values=SetMaskMetric.LABEL_VALUES unchanged - this metric
    never falls back to OTHER_LABEL.

    NO MASKING MOD, unlike every other wrapper in this module or in
    gwent_one/masked_field_dojos.py: SetMaskMetric is deliberately NOT
    a MaskedFieldMetric subclass (see its own module docstring) -
    cardset_tags was excluded from raw_content at ingestion time, so it
    was never part of a card's embedding input in the first place.
    There is nothing to mask; injecting a MaskTargetKeyMod here would
    only add a foreign "cardset_tags": "[MASK]" key that no real
    raw_content ever has, for no actual leak it would be preventing.
    """

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or SetMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor("label"),
            label_values=SetMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
