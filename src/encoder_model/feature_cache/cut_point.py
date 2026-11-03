"""CutPoint: where a feature cache cuts the card pipeline. The one source
of truth for the cut: the fingerprint holds it, the store's row kind and
the pipeline's PipelineCut are both derived from it.

A discriminated union:
- PooledCutPoint: after the pooling, one vector per text.
- TokenCutPoint(layer): token states, one row per real token. layer is
  None for the text encoder's whole output (any TextEncoder), or k for
  the states after transformer layer k of a LayeredTextEncoder (its
  layers above k run live and may train).
"""

from dataclasses import dataclass

import torch

from src.encoder_model.feature_cache.feature_row_kind import (
    POOLED_ROWS,
    TOKEN_ROWS,
    FeatureRowKind,
)
from src.encoder_model.feature_cache.ragged_token_states import RaggedTokenStates


@dataclass(frozen=True)
class PooledCutPoint:
    """After the pooling: the text encoder, enrichment and pooling are
    cached."""

    @property
    def row_kind(self) -> FeatureRowKind[torch.Tensor]:
        """POOLED_ROWS. Inputs: none. Output: FeatureRowKind. Side effects:
        none. Exceptions: none."""
        return POOLED_ROWS

    def describe(self) -> str:
        """The label "pooled". Inputs: none. Output: str. Side effects: none.
        Exceptions: none."""
        return "pooled"


@dataclass(frozen=True)
class TokenCutPoint:
    """Token states: after transformer layer `layer` of the text encoder,
    or (layer None) its whole output.

    layer: None, or >= 0 (0 is the token embeddings' output).
    """

    layer: int | None = None

    def __post_init__(self) -> None:
        """Inputs: none (checks self). Output: none. Side effects: none.
        Exceptions: ValueError for a negative layer."""
        if self.layer is not None and self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")

    @property
    def row_kind(self) -> FeatureRowKind[RaggedTokenStates]:
        """TOKEN_ROWS. Inputs: none. Output: FeatureRowKind. Side effects:
        none. Exceptions: none."""
        return TOKEN_ROWS

    def describe(self) -> str:
        """The label "tokens" or "tokens@<layer>". Inputs: none. Output: str.
        Side effects: none. Exceptions: none.

        Example:
            >>> TokenCutPoint(20).describe()
            'tokens@20'
        """
        return "tokens" if self.layer is None else f"tokens@{self.layer}"


CutPoint = PooledCutPoint | TokenCutPoint
