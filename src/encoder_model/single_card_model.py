"""SingleCardModel: every card embedded on its own (see CardEncoderModel)."""

from typing import List

import torch

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.card_encoder_model import CardEncoderModel
from src.schema.card import GenericCard


class SingleCardModel(CardEncoderModel):
    """Embeds every card on its own, with the injected CardEmbeddingPipeline.

    Cards never see each other, so embed_together and embed_apart are the
    same computation. All shape handling is in CardEncoderModel.
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

    def embed_together(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """Same as embed_apart: this model has no cross-card context.

        Inputs: cards (List[GenericCard]), non-empty.
        Output: List[Tensor], same length and order as cards.
        Side effects: none beyond autograd.
        Exceptions: as embed_apart.
        """
        return self.embed_apart(cards)

    def embed_apart(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One embedding per card, same order: CardEmbeddingPipeline.embed, split
        into one tensor per card.

        Inputs: cards (List[GenericCard]), from any onboarded game.
        Output: List[Tensor], same length and order as cards.
        Side effects: none directly (see CardEmbeddingPipeline.embed).
        Exceptions: whatever CardEmbeddingPipeline.embed raises.
        """
        return list(self.card_pipeline.embed(cards).unbind(0))
