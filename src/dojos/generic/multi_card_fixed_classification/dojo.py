"""Generic dojo for the (whole deck in, closed-vocabulary class out) task shape.

A GenericDojo (src/dojos/generic/generic_dojo.py) that supplies this cell's
decoder head and loss; everything else is the shared `Dojo` contract.
"""

from pathlib import Path
from typing import Sequence

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_card_fixed_classification.decoder_head import (
    MultiCardFixedClassificationDecoderHead,
)
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec


class MultiCardFixedClassificationDojo(GenericDojo):
    """Dojo for the (whole deck in, closed-vocabulary class out) task shape.

    Callers never build the decoder head or loss themselves; both are this
    cell's own detail. See GenericDojo for the shared constructor inputs,
    side effects and exceptions."""

    def __init__(
        self,
        path_to_training_data: Path,
        data_constructor: DataConstructor,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        label_values: Sequence[str],
        card_embedding_size: int,
        mod_pipeline: ModPipeline | None = None,
        pooler: EmbeddingPooler | None = None,
        rng_seed: int | None = None,
    ) -> None:
        label_values = list(label_values)
        self.card_embedding_size = card_embedding_size
        self.label_values = label_values
        super().__init__(
            path_to_training_data=path_to_training_data,
            data_constructor=data_constructor,
            card_lookup=card_lookup,
            holdout=holdout,
            decoder_head=MultiCardFixedClassificationDecoderHead(
                card_embedding_size, len(label_values), pooler=pooler
            ),
            loss_calculator=FixedClassificationLoss(label_values),
            mod_pipeline=mod_pipeline,
            rng_seed=rng_seed,
        )
