"""Thin per-metric wrappers over dominiontabs' two classification
metrics (src/data_refinement/metrics/dominiontabs/{type,set}_mask_metric.py).
Mirrors gwent_one/masked_field_dojos.py's convention: each wrapper adds
no behavior of its own, only configuration, pulling its paired
metric's own LABEL_VALUES/DEFAULT_OUTPUT_PATH ClassVars by reference.

MASKING MOD, train_only=False (TypeMaskDojo only - see SetMaskDojo's
own docstring for why it has none): the masked field must stay masked
on test/validation input too, not just training, or the model could
just read the answer off raw_content instead of predicting it - same
rationale as gwent_one's wrappers.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.dominiontabs.set_mask_metric import SetMaskMetric
from src.data_refinement.metrics.dominiontabs.type_mask_metric import TypeMaskMetric
from src.dojos.generic.data_constructors import MaskedFieldDataConstructor
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline


class TypeMaskDojo(SingleCardFixedClassificationDojo):
    """Card, primary type masked -> predicted primary type
    (TypeMaskMetric). label_values=TypeMaskMetric.LABEL_VALUES
    unchanged - OTHER_LABEL is already baked into that tuple."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or TypeMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=TypeMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [
                    MaskTargetKeyMod(
                        key=TypeMaskMetric.MASKED_FIELD[-1], train_only=False
                    )
                ]
            ),
            rng_seed=rng_seed,
        )


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
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or SetMaskMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=MaskedFieldDataConstructor(card_binder, "label"),
            label_values=SetMaskMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
