"""Turns one pooled vector per card into the card embedding.

EmbeddingHead is the last Strategy injected into a CardEmbeddingPipeline (text
encoder -> optional token enrichment -> token pooling -> embedding head):
which architecture projects the pooled vector (a linear projection, a
residual MLP, ...) is fully swappable here. Pooling tokens into that
vector is the pooling's job (pooling/), not the head's.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.encoder_model.fixed_width_stage import FixedWidthStage


class EmbeddingHead(FixedWidthStage, ABC):
    """Strategy interface: (batch, input_dim) pooled vectors in, one
    embedding per card out.

    Template method: each head's _embed does the architecture-specific
    step, and forward then applies a LayerNorm without learnable gain or
    bias. Every embedding therefore has mean 0 and variance 1 across its
    coordinates (L2 norm sqrt(output_dim)), so its scale cannot drift
    during training.

    input_dim: the pooled width this head reads (the pooling's
        output_dim, checked by CardEmbeddingPipeline).
    output_dim: the width of every embedding this head produces. Set once
        by each head's constructor, so a model can report its embedding
        size without running a forward pass.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Inputs: input_dim (the pooled width read), output_dim (the
            embedding width), both >= 1.
        Output: none (constructor). Side effects: builds the output norm.
        Exceptions: ValueError if either width is < 1."""
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        # No gain or bias: a learned bias would be one vector added to
        # every card, a shared direction the contrastive loss must undo
        self.output_norm = nn.LayerNorm(output_dim, elementwise_affine=False)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Inputs: pooled, (batch, input_dim), one pooled vector per card.
        Output: (batch, output_dim) tensor, one normalized embedding per
            card, same order as pooled.
        Side effects: none.
        Exceptions: none expected.
        """
        return self.output_norm(self._embed(pooled))

    @abstractmethod
    def _embed(self, pooled: torch.Tensor) -> torch.Tensor:
        """The head's own step, before normalization. Inputs and output
        as forward. Side effects: none. Exceptions: none expected."""
        raise NotImplementedError


class LinearEmbeddingHead(EmbeddingHead):
    """Baseline head: a single linear projection of the pooled vector."""

    def __init__(self, input_dim: int, card_embedding_size: int):
        """Inputs: input_dim (the pooling's output_dim), card_embedding_size.
        Output: none (constructor). Side effects: builds the projection.
        Exceptions: ValueError if either width is < 1.

        Example:
            >>> LinearEmbeddingHead(768, 256).output_dim
            256
        """
        super().__init__(input_dim=input_dim, output_dim=card_embedding_size)
        self.projection = nn.Linear(input_dim, card_embedding_size)

    def _embed(self, pooled: torch.Tensor) -> torch.Tensor:
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
    """Project the pooled vector into hidden_dim, then stack num_blocks
    residual bottleneck blocks before a final projection to
    card_embedding_size."""

    def __init__(
        self,
        input_dim: int,
        card_embedding_size: int,
        hidden_dim: int = 512,
        num_blocks: int = 3,
    ):
        """Inputs: input_dim (the pooling's output_dim), card_embedding_size,
            hidden_dim (the blocks' width), num_blocks (>= 0).
        Output: none (constructor). Side effects: builds the projections
            and blocks.
        Exceptions: ValueError if input_dim, card_embedding_size or
            hidden_dim < 1, or num_blocks < 0.

        Example:
            >>> ResidualMlpEmbeddingHead(768, 256, num_blocks=2).output_dim
            256
        """
        super().__init__(input_dim=input_dim, output_dim=card_embedding_size)
        if hidden_dim < 1 or num_blocks < 0:
            raise ValueError(
                f"hidden_dim must be >= 1 and num_blocks >= 0, got {hidden_dim}, {num_blocks}"
            )
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Linear(hidden_dim, card_embedding_size)

    def _embed(self, pooled: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(pooled)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden)
