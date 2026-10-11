"""TokenEnrichment: token states -> token states, between the text encoder
and the pooling.

The optional second stage of a CardEmbeddingPipeline (text encoder -> token
enrichment -> token pooling -> embedding head). A pipeline without one
passes the text encoder's tokens straight to the pooling, so there is no
identity subclass. Concrete enrichments (token whitening) live next to
this file.
"""

from abc import ABC, abstractmethod

from src.encoder_model.fixed_width_stage import FixedWidthStage
from src.encoder_model.text_encoder import TokenEncoding


class TokenEnrichment(FixedWidthStage, ABC):
    """Strategy interface: one TokenEncoding in, one out, with the same
    batch, seq_len and attention_mask; only the last dimension may change
    (input_dim -> output_dim). What an enrichment writes at padding
    positions does not matter, NaN included: every pooling zeroes padding
    before using the tokens (pooling/token_pooling.py, zero_padding).
    """

    @abstractmethod
    def forward(self, encoding: TokenEncoding) -> TokenEncoding:
        """
        Inputs: encoding, hidden_states (batch, seq_len, input_dim) and
            attention_mask (batch, seq_len).
        Output: a TokenEncoding with hidden_states (batch, seq_len,
            output_dim) and the same attention_mask.
        Side effects: none beyond autograd.
        Exceptions: none expected for a well-formed encoding.
        """
        raise NotImplementedError
