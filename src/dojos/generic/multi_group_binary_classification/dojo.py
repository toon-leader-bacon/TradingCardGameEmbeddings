"""Generic dojo for the (two card groups in, single logit out) task shape.

A GenericDojo (src/dojos/generic/generic_dojo.py) that supplies this
cell's decoder head, BceLoss and BinaryPriorCalibration (baseline: the
binary entropy of the TRAIN positive rate); everything else is the
shared `Dojo` contract. Labels are floats in {0.0, 1.0}.

Group order is load-bearing: input_shape_of() peeks group 0, so group 0
must never be empty; group 1 may be. For a symmetric pair task ("did
group 0 beat group 1") pass a ModPipeline holding a GroupSwapMod
(group_swap_mod.py) so TRAIN sees both orders.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_group_binary_classification.decoder_head import (
    MultiGroupBinaryClassificationDecoderHead,
)
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.loss.bce_loss import BceLoss
from src.dojos.loss.prior_baseline_calibrations import BinaryPriorCalibration
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec


class MultiGroupBinaryClassificationDojo(GenericDojo):
    """Dojo for the (two card groups in, single logit out) task shape.

    Callers never build the decoder head or loss themselves. See
    GenericDojo for the shared constructor inputs, side effects and
    exceptions."""

    def __init__(
        self,
        path_to_training_data: Path,
        data_constructor: DataConstructor,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        mod_pipeline: ModPipeline | None = None,
        pooler: EmbeddingPooler | None = None,
        deck_box: DeckBox | None = None,
        config: DojoConfig = DojoConfig(),
    ) -> None:
        self.card_embedding_size = card_embedding_size
        super().__init__(
            path_to_training_data=path_to_training_data,
            data_constructor=data_constructor,
            card_lookup=card_lookup,
            holdout=holdout,
            decoder_head=MultiGroupBinaryClassificationDecoderHead(
                card_embedding_size, pooler=pooler
            ),
            loss_calculator=BceLoss(),
            calibration=BinaryPriorCalibration(),
            mod_pipeline=mod_pipeline,
            deck_box=deck_box,
            config=config,
        )
