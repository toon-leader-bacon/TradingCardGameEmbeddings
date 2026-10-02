"""Generic dojo for the (single card in, closed-vocabulary class out) task shape.

A GenericDojo (src/dojos/generic/generic_dojo.py) that supplies this cell's
decoder head and loss; everything else is the shared `Dojo` contract.
"""

from pathlib import Path
from typing import Sequence

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.single_card_fixed_classification.decoder_head import (
    FixedClassificationDecoderHead,
)
from src.dojos.generic.single_card_fixed_classification.loss_spec import (
    FIXED_CLASSIFICATION_LOSS_SPEC,
    LossSpec,
)
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec


class SingleCardFixedClassificationDojo(GenericDojo):
    """Dojo for the (single card in, closed-vocabulary class out) task shape.

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
        deck_box: DeckBox | None = None,
        config: DojoConfig = DojoConfig(),
        # The loss and its matching calibration, as one value
        loss_spec: LossSpec = FIXED_CLASSIFICATION_LOSS_SPEC,
    ) -> None:
        label_values = list(label_values)
        self.card_embedding_size = card_embedding_size
        self.label_values = label_values
        super().__init__(
            path_to_training_data=path_to_training_data,
            data_constructor=data_constructor,
            card_lookup=card_lookup,
            holdout=holdout,
            decoder_head=FixedClassificationDecoderHead(
                card_embedding_size, len(label_values)
            ),
            loss_calculator=loss_spec.loss_factory(label_values),
            calibration=loss_spec.calibration,
            mod_pipeline=mod_pipeline,
            deck_box=deck_box,
            config=config,
        )
