"""TransformerPooling: trainable transformer layers over the tokens, then
another pooling."""

import torch
import torch.nn as nn

from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.pooling.token_pooling import TokenPooling, zero_padding
from src.encoder_model.text_encoder import TokenEncoding


class TransformerPooling(TokenPooling):
    """A small trainable transformer encoder over a card's (frozen) token
    states, so tokens can mix before they are pooled, then the inner
    pooling. Composition: the inner pooling is injected, and limited to
    MeanPooling or AttentionPooling (no nesting). input_dim is the token
    width throughout; output_dim is the inner pooling's."""

    def __init__(
        self,
        input_dim: int,
        inner: MeanPooling | AttentionPooling,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Inputs: input_dim (token width, the layers' d_model), inner (the
            pooling after the layers), num_layers (>= 1), num_heads (must
            divide input_dim), ffn_dim (each layer's feed-forward width;
            None means 2 * input_dim), dropout (inside each layer).
        Output: none (constructor).
        Side effects: builds the pre-norm transformer layers.
        Exceptions: ValueError if inner.input_dim != input_dim, num_layers
            < 1, num_heads < 1, or input_dim % num_heads != 0.

        Example:
            >>> TransformerPooling(768, inner=AttentionPooling(768)).output_dim
            768
        """
        # Validate the layer sizes and the inner pooling's width
        _check_transformer_sizes(input_dim, inner, num_layers, num_heads)
        super().__init__(input_dim=input_dim, output_dim=inner.output_dim)
        # Pre-norm layers on unpadded-or-masked sequences; nested tensors
        # off, as in MultiCardModel (pre-norm layers cannot use them)
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim if ffn_dim is not None else 2 * input_dim,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.inner = inner

    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        """See TokenPooling.forward.

        Example:
            >>> TransformerPooling(8, inner=MeanPooling(8), num_heads=2)(encoding).shape
            torch.Size([batch, 8])
        """
        # Let real tokens attend to each other, never to padding. Padding
        # is zeroed too: a masked key still multiplies its value by 0, and
        # 0 * NaN would leak into every real token
        padding = encoding.attention_mask == 0
        mixed = self.layers(zero_padding(encoding), src_key_padding_mask=padding)
        # Pool the mixed tokens with the same mask
        return self.inner(
            TokenEncoding(hidden_states=mixed, attention_mask=encoding.attention_mask)
        )


def _check_transformer_sizes(
    input_dim: int, inner: TokenPooling, num_layers: int, num_heads: int
) -> None:
    """Private helper - single consumer is TransformerPooling.__init__.

    Inputs: see TransformerPooling.__init__.
    Output: none. Side effects: none.
    Exceptions: ValueError naming the first size that does not fit.
    """
    if inner.input_dim != input_dim:
        raise ValueError(
            f"inner pooling reads width {inner.input_dim}, but the layers output {input_dim}"
        )
    if num_layers < 1 or num_heads < 1:
        raise ValueError(
            f"num_layers and num_heads must be >= 1, got {num_layers} and {num_heads}"
        )
    if input_dim % num_heads != 0:
        raise ValueError(f"num_heads {num_heads} does not divide input_dim {input_dim}")
