"""SingleCardModel: every card embedded on its own (see CardEncoderModel)."""

from typing import List

import torch

from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.embedding_head import EmbeddingHead
from src.encoder_model.text_encoder import TextEncoder
from src.schema.card import GenericCard


class SingleCardModel(CardEncoderModel):
    """Embeds every card on its own: text encoder, then embedding head.

    Cards never see each other, so embed_together and embed_apart are the
    same computation. All shape handling is in CardEncoderModel.
    """

    def __init__(self, text_encoder: TextEncoder, embedding_head: EmbeddingHead):
        super().__init__()
        # Strategy + dependency injection: the caller hands in both
        # collaborators, so swapping the text encoder (pretrained today,
        # maybe one trained from scratch later) or the head architecture
        # (linear, residual MLP, ...) never requires changing this class.
        self.text_encoder: TextEncoder = text_encoder
        self.embedding_head: EmbeddingHead = embedding_head

    @property
    def embedding_dim(self) -> int:
        """The injected head's output_dim. Inputs: none. Output: int.
        Side effects: none. Exceptions: none."""
        return self.embedding_head.output_dim

    def embed_together(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """Same as embed_apart: this model has no cross-card context.

        Inputs: cards (List[GenericCard]), non-empty.
        Output: List[Tensor], same length and order as cards.
        Side effects: none beyond autograd.
        Exceptions: as embed_apart.
        """
        return self.embed_apart(cards)

    def embed_apart(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One embedding per card, same order. Every card is serialized to
        text, the texts are encoded in one batch, and the injected head
        pools and projects them.

        Inputs: cards (List[GenericCard]), from any onboarded game.
        Output: List[Tensor], same length and order as cards.
        Side effects: none directly (the injected TextEncoder may
            accumulate gradients if it isn't frozen).
        Exceptions: whatever the injected TextEncoder/EmbeddingHead raise.
        """
        texts = [serialize_card_to_json(card) for card in cards]
        encoding = self.text_encoder.encode(texts)
        embeddings = self.embedding_head(encoding)
        return list(embeddings.unbind(0))
