"""FixedWidthStage: a pipeline stage whose input and output widths are fixed
at construction.

The shared base of TokenEnrichment, TokenPooling and EmbeddingHead, so
CardEmbeddingPipeline can check that adjacent stages chain without running
a forward pass. Each subclass states only its own forward contract.
"""

import torch.nn as nn


class FixedWidthStage(nn.Module):
    """input_dim: the width this stage reads. output_dim: the width it
    returns. Both validated (>= 1) and read-only."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Inputs: input_dim, output_dim (both >= 1).
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if either width is < 1.

        Example:
            >>> FixedWidthStage(768, 256).output_dim
            256
        """
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError(
                f"widths must be >= 1, got input_dim={input_dim}, output_dim={output_dim}"
            )
        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def input_dim(self) -> int:
        """Read-only: the width this stage reads. Inputs: none.
        Output: int. Side effects: none. Exceptions: none."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Read-only: the width this stage returns. Inputs: none.
        Output: int. Side effects: none. Exceptions: none."""
        return self._output_dim
