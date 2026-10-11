"""MultiCardModel: card embeddings contextualized within a group by
self-attention (see CardEncoderModel)."""

from typing import List

import torch
import torch.nn as nn

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.card_encoder_model import CardEncoderModel
from src.schema.card import GenericCard


class MultiCardModel(CardEncoderModel):
    """Each card is embedded on its own by the CardEmbeddingPipeline. A Transformer
    encoder then lets every card in a group attend to the others.

    embed_together runs self-attention over the whole group. embed_apart
    runs it over length-one sequences, so an unrelated card in the same
    batch never influences another's embedding. All shape handling is in
    CardEncoderModel.
    """

    def __init__(
        self,
        card_pipeline: CardEmbeddingPipeline,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        """Inputs: card_pipeline (its output_dim is the model's embedding
            width and self-attention's d_model),
            num_heads, num_layers, ffn_dim (each layer's feed-forward
            width), dropout, norm_first (pre-norm: LayerNorm before
            attention and feed-forward rather than after). The defaults
            are torch's TransformerEncoderLayer defaults; the activation
            is ReLU.
        Side effects: builds the self-attention layers, which end in a
            LayerNorm without gain or bias (L2 norm sqrt(embedding_dim)).
        Exceptions: AssertionError from torch if card_pipeline.output_dim
            is not divisible by num_heads.
        """
        super().__init__()
        # Dependency injection, the same seam as SingleCardModel: the
        # card pipeline turns each card into a base embedding
        # (card_pipeline.output_dim wide). self_attention below then lets
        # every card in a group attend to every other card, at that same
        # width: the pipeline is the one source of the embedding size.
        self.card_pipeline: CardEmbeddingPipeline = card_pipeline
        width = card_pipeline.output_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True,
        )
        # Nested tensors are a padding speed-up; groups here are unpadded,
        # and torch warns that pre-norm layers cannot use them anyway. The
        # final LayerNorm (no gain or bias, as in EmbeddingHead) closes the
        # stack: pre-norm layers leave their residual stream unnormalized
        self.self_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(width, elementwise_affine=False),
            enable_nested_tensor=False,
        )

    @property
    def embedding_dim(self) -> int:
        """The card pipeline's output_dim (self-attention keeps that
        width). Inputs: none. Output: int. Side effects: none.
        Exceptions: none."""
        return self.card_pipeline.output_dim

    def embed_together(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One contextualized embedding per card, same order. Every card
        attends to every other card in `cards`.

        Inputs: cards (List[GenericCard]), one group, non-empty.
        Output: List[Tensor] of shape (embedding_dim,), same order.
        Side effects: none beyond autograd.
        Exceptions: whatever CardEmbeddingPipeline.embed or the attention raise.
        """
        base = self.card_pipeline.embed(cards)  # (num_cards, embedding_dim)
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
        Output: List[Tensor] of shape (embedding_dim,), same order.
        Side effects: none beyond autograd.
        Exceptions: whatever CardEmbeddingPipeline.embed or the attention raise.
        """
        base = self.card_pipeline.embed(cards)  # (num_cards, embedding_dim)
        # Each card is its own sequence: (num_cards, 1, embedding_dim)
        isolated = self.self_attention(base.unsqueeze(1)).squeeze(1)
        return list(isolated.unbind(0))
