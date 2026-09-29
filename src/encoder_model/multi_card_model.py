"""MultiCardModel: card embeddings contextualized within a group by
self-attention (see CardEncoderModel)."""

from typing import List

import torch
import torch.nn as nn

from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.embedding_head import EmbeddingHead
from src.encoder_model.text_encoder import TextEncoder
from src.schema.card import GenericCard


class MultiCardModel(CardEncoderModel):
    """Each card is embedded by the text encoder and embedding head. A
    Transformer encoder then lets every card in a group attend to the
    others.

    embed_together runs self-attention over the whole group. embed_apart
    runs it over length-one sequences, so an unrelated card in the same
    batch never influences another's embedding. All shape handling is in
    CardEncoderModel.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        embedding_head: EmbeddingHead,
        card_embedding_size: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        # Strategy + dependency injection, the same seam as SingleCardModel:
        # text_encoder/embedding_head turn each card into a base embedding
        # (card_embedding_size wide). self_attention below then lets every
        # card in a group attend to every other card.
        self.text_encoder: TextEncoder = text_encoder
        self.embedding_head: EmbeddingHead = embedding_head
        self.card_embedding_size = card_embedding_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_embedding_size,
            nhead=num_heads,
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def embed_together(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One contextualized embedding per card, same order. Every card
        attends to every other card in `cards`.

        Inputs: cards (List[GenericCard]), one group, non-empty.
        Output: List[Tensor] of shape (card_embedding_size,), same order.
        Side effects: none beyond autograd.
        Exceptions: whatever the text encoder, head or attention raise.
        """
        base = self._base_embeddings(cards)  # (num_cards, card_embedding_size)
        # One group = one sequence: add a batch dim of 1, then drop it
        contextualized = self.self_attention(base.unsqueeze(0)).squeeze(0)
        return list(contextualized.unbind(0))

    def embed_apart(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One embedding per card, same order. Each card attends only to
        itself.

        The cards are encoded as one batch. Self-attention then runs over a
        batch of length-one sequences, so the transform is the same as for
        one card alone, without one text-encode call per card.

        Inputs: cards (List[GenericCard]), unrelated cards, non-empty.
        Output: List[Tensor] of shape (card_embedding_size,), same order.
        Side effects: none beyond autograd.
        Exceptions: whatever the text encoder, head or attention raise.
        """
        base = self._base_embeddings(cards)  # (num_cards, card_embedding_size)
        # Each card is its own sequence: (num_cards, 1, card_embedding_size)
        isolated = self.self_attention(base.unsqueeze(1)).squeeze(1)
        return list(isolated.unbind(0))

    def _base_embeddings(self, cards: List[GenericCard]) -> torch.Tensor:
        """Text-encode and head-project every card in one batch, before
        self-attention.

        Inputs: cards (List[GenericCard]).
        Output: Tensor (len(cards), card_embedding_size).
        Side effects: none beyond autograd.
        Exceptions: whatever the text encoder or head raise.
        """
        texts = [serialize_card_to_json(card) for card in cards]
        return self.embedding_head(self.text_encoder.encode(texts))
