"""PipelineCut: how a CardEmbeddingPipeline gets from texts to pooled
vectors, split in two at the point a feature cache stands in for.

Strategy, private to CardEmbeddingPipeline (its single consumer). The
part before the cut is what the cache stores; the part after runs on
every call:

- PooledCut: text encoder -> enrichment -> pooling come before the cut
  (one vector per text); nothing but the head runs after. It is also
  the pipeline's path without a cache.
- EncoderOutputCut: the text encoder's whole output (token states, one
  row per token, RaggedTokenStates) comes before the cut; any
  TextEncoder. Enrichment and pooling run live, and may train.
- LayerCut(k): a LayeredTextEncoder's states after layer k come before
  the cut; its layers above k, the enrichment and the pooling run live,
  and may train.

build_pipeline_cut derives the cut from the cache's CutPoint, so a cache
and the cut that reads it always agree.
"""

from typing import Any, Protocol, TypeVar

import torch
import torch.nn as nn

from src.encoder_model.enrichment.token_enrichment import TokenEnrichment
from src.encoder_model.feature_cache.cut_point import (
    CutPoint,
    PooledCutPoint,
    TokenCutPoint,
)
from src.encoder_model.feature_cache.ragged_token_states import RaggedTokenStates
from src.encoder_model.pooling.token_pooling import TokenPooling
from src.encoder_model.text_encoder import (
    LayeredTextEncoder,
    TextEncoder,
    TokenEncoding,
)

F = TypeVar("F")


class PipelineCut(Protocol[F]):
    """Strategy: the two halves of a pipeline, up to the pooled vector."""

    def encode_before_cut(self, texts: list[str]) -> F:
        """The part before the cut, live: what a cache stores for texts.
        Inputs: texts (non-empty). Output: F, one entry per text.
        Side effects: none beyond the stages' own. Exceptions: whatever the
        stages raise."""
        ...

    def pool_after_cut(self, features: F) -> torch.Tensor:
        """The part after the cut, up to the pooling's output.
        Inputs: features (float32, as encode_before_cut or the cache
        returns them). Output: (texts, pooling.output_dim) Tensor, with
        autograd through any trainable stage. Side effects: none beyond
        autograd. Exceptions: whatever the stages raise."""
        ...

    def check_frozen(self) -> None:
        """Refuse a cut whose cached part could change.
        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming the first stage before the cut with
        a parameter that requires grad."""
        ...

    def keep_cached_stages_in_eval(self) -> None:
        """After train(mode): return the stages before the cut to eval, so
        cached features are always eval-mode features. Inputs: none.
        Output: none. Side effects: sets those stages' modes.
        Exceptions: none."""
        ...


class PooledCut:
    """PipelineCut[Tensor]: everything before the head is before the cut."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        enrichment: TokenEnrichment | None,
        pooling: TokenPooling,
    ) -> None:
        """Inputs: the pipeline's stages before the head. Output: none
        (constructor). Side effects: none. Exceptions: none.

        Example:
            >>> PooledCut(encoder, None, MeanPooling(768))
        """
        self._text_encoder = text_encoder
        self._enrichment = enrichment
        self._pooling = pooling

    def encode_before_cut(self, texts: list[str]) -> torch.Tensor:
        """encode -> enrich (when set) -> pool. See PipelineCut.

        Example:
            >>> cut.encode_before_cut(["{...}"]).shape
            torch.Size([1, 768])
        """
        encoding = self._text_encoder.encode(texts)
        return enrich_and_pool(encoding, self._enrichment, self._pooling)

    def pool_after_cut(self, features: torch.Tensor) -> torch.Tensor:
        """features themselves: they are already pooled. See PipelineCut.

        Example:
            >>> cut.pool_after_cut(pooled) is pooled
            True
        """
        return features

    def check_frozen(self) -> None:
        """Every stage before the head frozen. See PipelineCut.

        Example:
            >>> PooledCut(frozen_encoder, None, MeanPooling(768)).check_frozen()
        """
        stages: list[tuple[str, nn.Module | None]] = [
            ("text encoder", self._text_encoder),
            ("enrichment", self._enrichment),
            ("pooling", self._pooling),
        ]
        _require_frozen(stages, "every stage before the head")

    def keep_cached_stages_in_eval(self) -> None:
        """text encoder, enrichment and pooling to eval. See PipelineCut."""
        for stage in (self._text_encoder, self._enrichment, self._pooling):
            if stage is not None:
                stage.eval()


class EncoderOutputCut:
    """PipelineCut[RaggedTokenStates]: the text encoder's whole output is
    before the cut; enrichment and pooling run live. Works with any
    TextEncoder."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        enrichment: TokenEnrichment | None,
        pooling: TokenPooling,
    ) -> None:
        """Inputs: the pipeline's stages before the head. Output: none
        (constructor). Side effects: none. Exceptions: none.

        Example:
            >>> EncoderOutputCut(encoder, None, AttentionPooling(768))
        """
        self._text_encoder = text_encoder
        self._enrichment = enrichment
        self._pooling = pooling

    def encode_before_cut(self, texts: list[str]) -> RaggedTokenStates:
        """The encoder's token states, packed without padding. See
        PipelineCut.

        Example:
            >>> cut.encode_before_cut(["{...}"]).lengths.tolist()
            [37]
        """
        return RaggedTokenStates.from_encoding(self._text_encoder.encode(texts))

    def pool_after_cut(self, features: RaggedTokenStates) -> torch.Tensor:
        """Pad, enrich, pool. See PipelineCut.

        Example:
            >>> cut.pool_after_cut(states).shape
            torch.Size([1, 768])
        """
        return enrich_and_pool(features.to_encoding(), self._enrichment, self._pooling)

    def check_frozen(self) -> None:
        """The whole text encoder frozen. See PipelineCut.

        Example:
            >>> EncoderOutputCut(frozen_encoder, None, AttentionPooling(768)).check_frozen()
        """
        _require_frozen([("text encoder", self._text_encoder)], "the text encoder")

    def keep_cached_stages_in_eval(self) -> None:
        """The text encoder to eval. See PipelineCut."""
        self._text_encoder.eval()


class LayerCut:
    """PipelineCut[RaggedTokenStates]: a LayeredTextEncoder's states after
    layer k are before the cut; its layers above k, the enrichment and
    the pooling run live, and may train."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        layer: int,
        enrichment: TokenEnrichment | None,
        pooling: TokenPooling,
    ) -> None:
        """Inputs: text_encoder (must be a LayeredTextEncoder), layer (0 to
            its layer_count), enrichment, pooling.
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if text_encoder cannot be split, or layer
            is out of range.

        Example:
            >>> encoder = load_text_encoder("answerdotai/ModernBERT-base")
            >>> LayerCut(encoder, 20, None, MeanPooling(768))
        """
        self._text_encoder = _require_layered(text_encoder, layer)
        self._layer = layer
        self._enrichment = enrichment
        self._pooling = pooling

    def encode_before_cut(self, texts: list[str]) -> RaggedTokenStates:
        """The states after `layer` layers, packed without padding. See
        PipelineCut.

        Example:
            >>> cut.encode_before_cut(["{...}"]).lengths.tolist()
            [37]
        """
        encoding = self._text_encoder.encode_through(texts, self._layer)
        return RaggedTokenStates.from_encoding(encoding)

    def pool_after_cut(self, features: RaggedTokenStates) -> torch.Tensor:
        """Pad, run the encoder's layers above the cut, enrich, pool. See
        PipelineCut.

        Example:
            >>> cut.pool_after_cut(states).shape
            torch.Size([1, 768])
        """
        encoding = self._text_encoder.encode_from(features.to_encoding(), self._layer)
        return enrich_and_pool(encoding, self._enrichment, self._pooling)

    def check_frozen(self) -> None:
        """The encoder frozen through the cut; the stages after it may
        train. See PipelineCut.

        Example:
            >>> LayerCut(encoder, 20, None, MeanPooling(768)).check_frozen()
        """
        if not self._text_encoder.is_frozen_through(self._layer):
            raise ValueError(
                f"the text encoder trains a layer at or below {self._layer}; a "
                "token cut there needs everything below it frozen"
            )

    def keep_cached_stages_in_eval(self) -> None:
        """The encoder's frozen layers to eval (LayeredTextEncoder's
        keep_frozen_layers_in_eval). See PipelineCut."""
        self._text_encoder.keep_frozen_layers_in_eval()


def build_pipeline_cut(
    cut_point: CutPoint,
    text_encoder: TextEncoder,
    enrichment: TokenEnrichment | None,
    pooling: TokenPooling,
) -> PipelineCut[Any]:
    """The PipelineCut a cache's CutPoint names, over the pipeline's stages.

    Factory Method. Inputs: cut_point, the stages before the head.
    Output: PooledCut, EncoderOutputCut (a token cut without a layer) or
        LayerCut. Side effects: none.
    Exceptions: ValueError from LayerCut for an encoder it cannot split.

    Example:
        >>> build_pipeline_cut(TokenCutPoint(20), encoder, None, MeanPooling(768))
    """
    if isinstance(cut_point, PooledCutPoint):
        return PooledCut(text_encoder, enrichment, pooling)
    assert isinstance(cut_point, TokenCutPoint)  # CutPoint has two variants
    if cut_point.layer is None:
        return EncoderOutputCut(text_encoder, enrichment, pooling)
    return LayerCut(text_encoder, cut_point.layer, enrichment, pooling)


def enrich_and_pool(
    encoding: TokenEncoding,
    enrichment: TokenEnrichment | None,
    pooling: TokenPooling,
) -> torch.Tensor:
    """The token stages every path ends with: enrich (when set), then pool.

    Inputs: encoding, enrichment, pooling.
    Output: (texts, pooling.output_dim) Tensor.
    Side effects: none beyond autograd. Exceptions: the stages'.

    Example:
        >>> enrich_and_pool(encoding, None, MeanPooling(768)).shape
        torch.Size([2, 768])
    """
    if enrichment is not None:
        encoding = enrichment(encoding)
    return pooling(encoding)


def _require_layered(text_encoder: TextEncoder, layer: int) -> LayeredTextEncoder:
    """Parse the text encoder into one a token cut can split.

    Private helper - single consumer is LayerCut.__init__.
    Inputs: text_encoder, layer. Output: the same encoder, typed.
    Side effects: none.
    Exceptions: ValueError if it is not a LayeredTextEncoder, or layer is
        outside [0, layer_count].
    """
    if not isinstance(text_encoder, LayeredTextEncoder):
        raise ValueError(
            f"a token cut at layer {layer} needs a text encoder that can be split "
            f"(a LayeredTextEncoder), not {type(text_encoder).__name__}"
        )
    if not 0 <= layer <= text_encoder.layer_count:
        raise ValueError(
            f"cut layer must be in [0, {text_encoder.layer_count}], got {layer}"
        )
    return text_encoder


def _require_frozen(stages: list[tuple[str, nn.Module | None]], scope: str) -> None:
    """Private helper - consumers are the cuts' check_frozen.

    Inputs: stages ((name, module or None) in order), scope (what a cache
        needs frozen, for the message).
    Output: none. Side effects: none.
    Exceptions: ValueError naming the first stage with a parameter that
        requires grad.
    """
    for name, stage in stages:
        if stage is None:
            continue
        if any(parameter.requires_grad for parameter in stage.parameters()):
            raise ValueError(
                f"{name} has trainable parameters; a feature cache needs {scope} "
                "frozen"
            )
