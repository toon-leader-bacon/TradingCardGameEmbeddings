"""AttentionPooling: learned queries, each a weighted average of the tokens."""

import math

import torch

from src.encoder_model.pooling.token_pooling import TokenPooling, zero_padding
from src.encoder_model.text_encoder import TokenEncoding


class AttentionPooling(TokenPooling):
    """num_queries learned query vectors. Each scores every real token
    (scaled dot product), softmaxes over the tokens, and takes the weighted
    average; the num_queries averages are concatenated, so
    output_dim = num_queries * input_dim. One query is the classic
    single-vector attention pooling; several let different queries pick
    out different parts of a card (cost, type line, rules text)."""

    def __init__(self, input_dim: int, num_queries: int = 1) -> None:
        """Inputs: input_dim (>= 1), the token width; num_queries (>= 1).
        Output: none (constructor).
        Side effects: creates the (num_queries, input_dim) query parameter.
        Exceptions: ValueError if num_queries < 1 or input_dim < 1.

        Example:
            >>> AttentionPooling(768, num_queries=4).output_dim
            3072
        """
        # Validate num_queries before the base computes output_dim from it
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        super().__init__(input_dim=input_dim, output_dim=num_queries * input_dim)
        self.num_queries = num_queries
        self.queries = torch.nn.Parameter(torch.randn(num_queries, input_dim))

    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        """See TokenPooling.forward.

        Example:
            >>> AttentionPooling(8, num_queries=2)(encoding).shape
            torch.Size([batch, 16])
        """
        # Padding is zeroed first, so whatever it held cannot leak in
        hidden_states = zero_padding(encoding)
        # Score every token against every query: (batch, seq_len, num_queries)
        scores = self._token_scores(hidden_states)
        # Padding can never be attended to
        weights = self._masked_softmax(scores, encoding.attention_mask)
        # Each query's weighted average: (batch, num_queries, input_dim)
        pooled = torch.einsum("bsq,bsd->bqd", weights, hidden_states)
        # Concatenate the queries' averages
        return pooled.flatten(start_dim=1)

    def _token_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Inputs: hidden_states (batch, seq_len, input_dim).
        Output: (batch, seq_len, num_queries), dot products scaled by
            1 / sqrt(input_dim) so the softmax starts out soft.
        Side effects: none. Exceptions: none."""
        queries = self.queries.to(hidden_states.dtype)
        return (hidden_states @ queries.T) / math.sqrt(self.input_dim)

    @staticmethod
    def _masked_softmax(
        scores: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Inputs: scores (batch, seq_len, num_queries), attention_mask
            (batch, seq_len).
        Output: softmax over seq_len with padding positions at weight 0.
        Side effects: none. Exceptions: none."""
        padding = (attention_mask == 0).unsqueeze(-1)
        return torch.softmax(scores.masked_fill(padding, float("-inf")), dim=1)
