"""SingleCardModel: every card embedded on its own (see CardEncoderModel)."""

import torch

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.card_encoder_model import CardEncoderModel


class SingleCardModel(CardEncoderModel):
    """Embeds every card on its own, with the injected CardEmbeddingPipeline.

    Cards never see each other, so contextualize is the identity. All
    shape handling is in CardEncoderModel.
    """

    def __init__(self, card_pipeline: CardEmbeddingPipeline):
        """Inputs: card_pipeline (its four stages fix the architecture).
        Output: none (constructor). Side effects: none. Exceptions: none."""
        # Dependency injection: swapping any stage (text encoder,
        # enrichment, pooling, head) happens inside the CardEmbeddingPipeline, never
        # here
        super().__init__(card_pipeline)

    @property
    def embedding_dim(self) -> int:
        """The card pipeline's output_dim. Inputs: none. Output: int.
        Side effects: none. Exceptions: none."""
        return self.card_pipeline.output_dim

    def contextualize(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The identity: this model has no cross-card context.

        Inputs: cards, mask - see CardEncoderModel.contextualize.
        Output: cards, unchanged.
        Side effects: none. Exceptions: none.

        Example:
            >>> model.contextualize(cards, mask) is cards
            True
        """
        return cards
