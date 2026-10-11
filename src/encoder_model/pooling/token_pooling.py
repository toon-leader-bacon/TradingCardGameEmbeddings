"""TokenPooling: one TokenEncoding row (a card's token states) -> one vector.

The third stage of a CardEmbeddingPipeline (text encoder -> optional token
enrichment -> token pooling -> embedding head). A Strategy: which
pooling a model uses is injected, never chosen inside the model.
"""

from abc import ABC, abstractmethod

import torch

from src.encoder_model.fixed_width_stage import FixedWidthStage
from src.encoder_model.text_encoder import TokenEncoding


class TokenPooling(FixedWidthStage, ABC):
    """Strategy interface: (batch, seq_len, input_dim) token states plus
    their attention mask in, (batch, output_dim) out.

    Every pooling ignores padding (attention_mask == 0), whatever values
    sit there, NaN included. A row with no real token is a caller error:
    card texts are never empty.
    """

    @abstractmethod
    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        """
        Inputs: encoding, hidden_states (batch, seq_len, input_dim) and
            attention_mask (batch, seq_len).
        Output: (batch, output_dim) tensor, one pooled vector per row, same
            order.
        Side effects: none beyond autograd.
        Exceptions: none expected for a well-formed encoding.
        """
        raise NotImplementedError


def zero_padding(encoding: TokenEncoding) -> torch.Tensor:
    """encoding.hidden_states with every padding position set to exactly 0,
    selected rather than multiplied, so NaN or inf at padding cannot leak
    into a sum (0 * NaN is NaN). Shared by the poolings that sum tokens.

    Inputs: encoding (TokenEncoding).
    Output: tensor shaped like encoding.hidden_states.
    Side effects: none. Exceptions: none.

    Example:
        >>> zero_padding(encoding)[0, 3]   # a padding position
        tensor([0., 0., ...])
    """
    padding = (encoding.attention_mask == 0).unsqueeze(-1)
    return encoding.hidden_states.masked_fill(padding, 0.0)
