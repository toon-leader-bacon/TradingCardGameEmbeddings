"""Generic dojo for the (whole deck in, scalar regression out) task shape.

A GenericDojo (src/dojos/generic/generic_dojo.py) that supplies this cell's
decoder head and loss; everything else is the shared `Dojo` contract.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_card_regression.decoder_head import (
    MultiCardRegressionDecoderHead,
)
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.loss.mse_loss import MseLoss
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec


class MultiCardRegressionDojo(GenericDojo):
    """Dojo for the (whole deck in, scalar regression out) task shape.

    Callers never build the decoder head or loss themselves; both are this
    cell's own detail. See GenericDojo for the shared constructor inputs,
    side effects and exceptions."""

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
            decoder_head=MultiCardRegressionDecoderHead(
                card_embedding_size, pooler=pooler
            ),
            loss_calculator=MseLoss(),
            mod_pipeline=mod_pipeline,
            deck_box=deck_box,
            config=config,
        )
