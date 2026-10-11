"""MeanPooling: the average of a card's real token states."""

import torch

from src.encoder_model.pooling.token_pooling import TokenPooling, zero_padding
from src.encoder_model.text_encoder import TokenEncoding


class MeanPooling(TokenPooling):
    """Average every real (non-padding) token. No parameters, so the same
    tokens always pool to the same vector. output_dim == input_dim."""

    def __init__(self, input_dim: int) -> None:
        """Inputs: input_dim (>= 1), the token width.
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if input_dim < 1.

        Example:
            >>> MeanPooling(768).output_dim
            768
        """
        super().__init__(input_dim=input_dim, output_dim=input_dim)

    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        """Masked mean over seq_len - see TokenPooling.forward.

        Example:
            >>> MeanPooling(8)(encoding).shape
            torch.Size([batch, 8])
        """
        # Sum the real tokens and divide by their count. The clamp only
        # avoids a division by zero; an all-padding row is a caller error
        summed = zero_padding(encoding).sum(dim=1)
        mask = encoding.attention_mask.to(summed.dtype)
        token_counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return summed / token_counts
