"""Thin wrapper over CardCharacterPredictionMetric
(src/data_refinement/metrics/sts_gg/card_character_prediction_metric.py).

A single SingleCardFixedClassificationDojo subclass - adds no behavior
of its own, only configuration, same convention as every other thin
wrapper in this plan (e.g. sts_gg/deck_label_dojos.py's CharacterDojo,
gwent_one/masked_field_dojos.py's wrappers): pulls
CardCharacterPredictionMetric.DEFAULT_OUTPUT_PATH by reference, injects
a CardCharacterPredictionDataConstructor, and passes
CharacterPredictionMetric.LABEL_VALUES (deck_label_metrics.py - the
deck-input mirror of this same prediction task, reused by reference
rather than re-declared, per CardCharacterPredictionMetric's own module
docstring) as label_values.

loss_factory=SoftClassificationLoss is what makes this dojo a soft-
label consumer of an otherwise-unmodified generic cell:
CardCharacterPredictionMetric's own rows carry a whole character
probability distribution per card, not one hard class, so this wrapper
is the one consumer of SingleCardFixedClassificationDojo's loss_factory
parameter (see plans/dojo_v2.md and that parameter's own docstring) -
every other wrapper of that same generic cell (gwent_one's 8) omits it
and gets FixedClassificationLoss's default behavior, unchanged.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts_gg.card_character_prediction_metric import (
    CardCharacterPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
)
from src.dojos.generic.data_constructors import CardCharacterPredictionDataConstructor
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.loss.soft_classification_loss import SoftClassificationLoss


class CardCharacterPredictionDojo(SingleCardFixedClassificationDojo):
    """Card -> character probability distribution (CardCharacterPredictionMetric).

    label_values=CharacterPredictionMetric.LABEL_VALUES unchanged - it
    already includes OTHER_LABEL (see that class's own docstring)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardCharacterPredictionMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardCharacterPredictionDataConstructor(card_binder),
            label_values=CharacterPredictionMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            loss_factory=SoftClassificationLoss,
            rng_seed=rng_seed,
        )
