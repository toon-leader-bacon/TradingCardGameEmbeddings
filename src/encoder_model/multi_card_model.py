"""MultiCardModel: card embeddings contextualized within a group by
self-attention (see CardEncoderModel)."""

import torch
import torch.nn as nn

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.card_encoder_model import CardEncoderModel


class MultiCardModel(CardEncoderModel):
    """Each card is embedded on its own by the CardEmbeddingPipeline. A Transformer
    encoder then lets every card in a group attend to the others.

    contextualize runs self-attention over every padded sequence at once,
    with a key padding mask, so a card never sees padding or another
    sequence's cards. A batch of single cards is a batch of length-one
    sequences. All shape handling is in CardEncoderModel.
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
        # Dependency injection, the same seam as SingleCardModel: the
        # card pipeline turns each card into a base embedding
        # (card_pipeline.output_dim wide). self_attention below then lets
        # every card in a group attend to every other card, at that same
        # width: the pipeline is the one source of the embedding size.
        super().__init__(card_pipeline)
        width = card_pipeline.output_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True,
        )
        # Nested tensors are a padding speed-up, but torch warns that
        # pre-norm layers cannot use them; the implement step measures
        # whether enabling them helps post-norm configs. The
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

    def contextualize(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """One self-attention call over every sequence; a card attends only
        to the real cards of its own sequence.

        Inputs: cards, mask - see CardEncoderModel.contextualize. The
            base class guarantees max_cards >= 1.
        Output: (sequences, max_cards, d). An empty sequence's rows are
            garbage that forward zeroes.
        Side effects: none beyond autograd.
        Exceptions: whatever the attention raises.

        Example:
            >>> model.contextualize(cards, mask).shape == cards.shape
            True
        """
        # A fully masked row makes attention produce NaN. Unmask slot 0 of
        # every empty sequence instead of selecting the non-empty rows:
        # boolean row selection would cost a device sync. The dummy slot
        # holds zeros, and only its own row reads it
        attention_mask = _with_dummy_slot_for_empty(mask)

        # One call; the key padding mask keeps padding out of every softmax
        return self.self_attention(cards, src_key_padding_mask=~attention_mask)


def _with_dummy_slot_for_empty(mask: torch.Tensor) -> torch.Tensor:
    """mask, with position 0 set True in every all-False row.

    Private helper - single caller is MultiCardModel.contextualize.
    Inputs: mask, (sequences, max_cards) bool, max_cards >= 1.
    Output: a new (sequences, max_cards) bool tensor on mask's device;
        rows with a real card are unchanged.
    Side effects: none (no device sync). Exceptions: none.
    """
    result = mask.clone()
    result[:, 0] |= ~mask.any(dim=1)
    return result
