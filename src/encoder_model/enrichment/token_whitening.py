"""TokenWhitening: every token mapped through a fitted whitening, before
pooling."""

import torch

from src.encoder_model.enrichment.token_enrichment import TokenEnrichment
from src.encoder_model.text_encoder import TokenEncoding
from src.encoder_model.whitening.whitening_stats import WhiteningStats


class TokenWhitening(TokenEnrichment):
    """(x - mean) @ projection on every token, with mean and projection
    from WhiteningStats held as non-trainable buffers (saved in the
    model's state dict with everything else). The frozen text encoder's
    tokens never change, so the fit never goes stale, whatever pooling
    follows. Output tokens are float32 whatever the input dtype. Padding
    is transformed too; every pooling masks it anyway."""

    mean: torch.Tensor
    projection: torch.Tensor

    def __init__(self, stats: WhiteningStats) -> None:
        """Inputs: stats (input_dim = the text encoder's width).
        Output: none (constructor).
        Side effects: registers the mean and projection buffers.
        Exceptions: none beyond FixedWidthStage's width check.

        Example:
            >>> TokenWhitening(WhiteningStats.load(path)).output_dim
            768
        """
        super().__init__(input_dim=stats.input_dim, output_dim=stats.output_dim)
        self.register_buffer("mean", stats.mean.clone())
        self.register_buffer("projection", stats.projection.clone())

    def forward(self, encoding: TokenEncoding) -> TokenEncoding:
        """See TokenEnrichment.forward.

        Example:
            >>> TokenWhitening(stats)(encoding).hidden_states.shape[-1]
            768
        """
        # Always in float32, autocast off: under autocast the matmul would
        # run in fp16, and the projection's large per-direction scales
        # (1 / sqrt of small eigenvalues) would amplify its rounding
        hidden = encoding.hidden_states
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            centered = hidden.float() - self.mean.float()
            whitened = centered @ self.projection.float()
        return TokenEncoding(
            hidden_states=whitened, attention_mask=encoding.attention_mask
        )
