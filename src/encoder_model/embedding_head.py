"""Turns one TextEncoder's per-token output into a single card embedding.

EmbeddingHead is a Strategy injected into SingleCardModel alongside
TextEncoder: which architecture handles the token-to-embedding step (a
mean-pooled linear projection, a residual MLP, a learned attention-pooling
layer, ...) is fully swappable here without SingleCardModel or
TextEncoder ever needing to change.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.encoder_model.text_encoder import TokenEncoding


def _mean_pool_tokens(encoding: TokenEncoding) -> torch.Tensor:
    """Mean-pool token hidden states over real (non-padding) tokens only.

    Inputs: encoding, a TextEncoder's output for a batch of texts.
    Output: (batch, hidden_dim) tensor, one pooled vector per input text.
    Side effects: none.
    Exceptions: none expected.
    """
    mask = encoding.attention_mask.unsqueeze(-1).to(encoding.hidden_states.dtype)
    summed = (encoding.hidden_states * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / token_counts


class EmbeddingHead(nn.Module, ABC):
    """Strategy interface: one TextEncoder output in, one embedding per
    card out."""

    @abstractmethod
    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        """
        Inputs: encoding, one TextEncoder's output for a batch of cards.
        Output: (batch, embed_dim) tensor, one embedding per input card,
            same order as encoding.
        Side effects: none.
        Exceptions: none expected.
        """
        raise NotImplementedError


class LinearEmbeddingHead(EmbeddingHead):
    """Baseline head: mean-pool tokens, then a single linear projection."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        pooled = _mean_pool_tokens(encoding)
        return self.projection(pooled)


class _ResidualBlock(nn.Module):
    """One Linear -> LayerNorm -> GELU bottleneck block with a residual
    add, the repeating unit ResidualMlpEmbeddingHead stacks."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.norm(self.linear(x)))


class ResidualMlpEmbeddingHead(EmbeddingHead):
    """Mean-pool tokens, project into hidden_dim, then stack num_blocks
    residual bottleneck blocks before a final projection to embed_dim."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        pooled = _mean_pool_tokens(encoding)
        hidden = self.input_projection(pooled)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden)


class AttentionPoolingEmbeddingHead(EmbeddingHead):
    """Pools tokens via a learned attention query instead of a uniform
    mean - lets the head learn which tokens (e.g. a card's rules text
    vs. its cost) matter most to the final embedding, rather than
    weighting every token equally the way _mean_pool_tokens does."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim))
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, encoding: TokenEncoding) -> torch.Tensor:
        # (batch, seq_len): how well each token matches the learned query.
        scores = encoding.hidden_states @ self.query
        scores = scores.masked_fill(encoding.attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (encoding.hidden_states * weights).sum(dim=1)
        return self.projection(pooled)
