"""ModernBertTextEncoder: a pretrained ModernBERT that can be split after
any layer, and whose top layers can train while the rest stay frozen.

A LayeredTextEncoder (text_encoder.py) through ModernBertLayerStack. Its
encode() always runs split, at the boundary between its frozen and
trainable layers, so there is one forward path for every use: plain
encoding, a token-level cache's lower half, and the live upper half.
"""

from typing import List

import torch
import torch.nn as nn
from transformers import AutoConfig

from src.encoder_model.modernbert_layer_stack import ModernBertLayerStack
from src.encoder_model.text_encoder import (
    DEFAULT_LENGTH_BUCKET_SIZE,
    DEFAULT_MAX_LENGTH,
    LayeredTextEncoder,
    PretrainedTextEncoder,
    TextEncoder,
    TokenEncoding,
)

# The Hugging Face model_type of the checkpoints this encoder can split
_MODERNBERT_MODEL_TYPE = "modernbert"


class ModernBertTextEncoder(PretrainedTextEncoder, LayeredTextEncoder):
    """A PretrainedTextEncoder for ModernBERT checkpoints, frozen except
    its top trainable_top_layers transformer layers and final norm. The
    base class's `trainable` (the whole model) is always False here:
    trainable_top_layers is this class's one trainability setting.

    The frozen layers stay in eval mode whatever train() is given; the
    trainable ones follow it.
    """

    def __init__(
        self,
        checkpoint: str,
        trainable_top_layers: int = 0,
        max_length: int = DEFAULT_MAX_LENGTH,
        model_kwargs: dict | None = None,
        length_bucket_size: int | None = DEFAULT_LENGTH_BUCKET_SIZE,
    ) -> None:
        """Inputs: checkpoint (a ModernBERT model id or path),
            trainable_top_layers (0 = wholly frozen, up to layer_count),
            max_length, model_kwargs, length_bucket_size (as
            PretrainedTextEncoder).
        Output: none (constructor).
        Side effects: loads the tokenizer and model; marks the top layers'
            and final norm's parameters trainable.
        Exceptions: ValueError if the model is not ModernBERT (see
            ModernBertLayerStack) or trainable_top_layers is outside [0,
            layer_count]; whatever from_pretrained raises.

        Example:
            >>> encoder = ModernBertTextEncoder(
            ...     "answerdotai/ModernBERT-base", trainable_top_layers=2
            ... )
            >>> encoder.is_frozen_through(20)
            True
        """
        super().__init__(
            checkpoint,
            trainable=False,
            max_length=max_length,
            model_kwargs=model_kwargs,
            length_bucket_size=length_bucket_size,
        )
        self._stack = ModernBertLayerStack(self.model)
        layer_count = self._stack.layer_count
        if not 0 <= trainable_top_layers <= layer_count:
            raise ValueError(
                f"trainable_top_layers must be in [0, {layer_count}], "
                f"got {trainable_top_layers}"
            )
        self.trainable_top_layers = trainable_top_layers
        # Thaw the top layers and the final norm; the rest stays frozen
        if trainable_top_layers:
            for parameter in self._stack.parameters_from(self._split_layer):
                parameter.requires_grad = True

    @property
    def layer_count(self) -> int:
        """The model's transformer layers. Inputs: none. Output: int.
        Side effects: none. Exceptions: none."""
        return self._stack.layer_count

    @property
    def _split_layer(self) -> int:
        """The first trainable layer (layer_count when all are frozen):
        where encode() splits. Inputs: none. Output: int. Side effects:
        none. Exceptions: none."""
        return self._stack.layer_count - self.trainable_top_layers

    def train(self, mode: bool = True) -> "ModernBertTextEncoder":
        """Set the mode: frozen layers stay in eval, the trainable top
        layers and final norm follow mode.

        Inputs: mode. Output: self. Side effects: sets submodule modes.
        Exceptions: none.

        Example:
            >>> encoder.train().model.layers[-1].training
            True
        """
        nn.Module.train(self, mode)
        self.keep_frozen_layers_in_eval()
        return self

    def keep_frozen_layers_in_eval(self) -> None:
        """See LayeredTextEncoder.keep_frozen_layers_in_eval: the
        embeddings and the layers below the trainable ones.

        Example:
            >>> encoder.keep_frozen_layers_in_eval()
        """
        self.model.embeddings.eval()
        for layer in self.model.layers[: self._split_layer]:
            layer.eval()
        if not self.trainable_top_layers:
            self.model.eval()

    def encode(self, texts: List[str]) -> TokenEncoding:
        """The lower, frozen layers then the upper ones: see
        TextEncoder.encode. With no trainable layers the upper half is
        only the final norm.

        Example:
            >>> encoder.encode(["{...}"]).hidden_states.shape
            torch.Size([1, 37, 768])
        """
        split = self._split_layer
        return self.encode_from(self.encode_through(texts, split), split)

    def encode_through(self, texts: List[str], layer: int) -> TokenEncoding:
        """See LayeredTextEncoder.encode_through: tokenized and bucketed as
        encode is, without autograd when those layers are frozen.

        Example:
            >>> encoder.encode_through(["{...}"], 20).hidden_states.shape
            torch.Size([1, 37, 768])
        """
        frozen = self.is_frozen_through(layer)

        def forward(
            input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
                return self._stack.run_through(input_ids, attention_mask, layer)

        return self._encode_with(texts, forward)

    def encode_from(self, encoding: TokenEncoding, layer: int) -> TokenEncoding:
        """See LayeredTextEncoder.encode_from: the remaining layers and the
        final norm, with autograd only when one of them trains.

        Example:
            >>> encoder.encode_from(encoder.encode_through(texts, 20), 20)
            TokenEncoding(hidden_states=..., attention_mask=...)
        """
        trains = any(p.requires_grad for p in self._stack.parameters_from(layer))
        with torch.set_grad_enabled(torch.is_grad_enabled() and trains):
            hidden_states = self._stack.run_from(
                encoding.hidden_states, encoding.attention_mask, layer
            )
        return TokenEncoding(hidden_states, encoding.attention_mask)

    def is_frozen_through(self, layer: int) -> bool:
        """See LayeredTextEncoder.is_frozen_through.

        Example:
            >>> ModernBertTextEncoder(checkpoint, trainable_top_layers=2).is_frozen_through(20)
            True
        """
        parameters = self._stack.parameters_through(layer)
        return not any(parameter.requires_grad for parameter in parameters)


def load_text_encoder(checkpoint: str, trainable_top_layers: int = 0) -> TextEncoder:
    """The text encoder for a pretrained checkpoint: a ModernBertTextEncoder
    (splittable, partly trainable) for a ModernBERT checkpoint, otherwise a
    frozen PretrainedTextEncoder.

    Factory Method: reads the checkpoint's config to pick the class.
    Inputs: checkpoint, trainable_top_layers (0 = frozen).
    Output: TextEncoder.
    Side effects: loads the checkpoint's config, tokenizer and weights
        (downloading them if needed).
    Exceptions: ValueError if trainable_top_layers > 0 for a checkpoint
        that is not ModernBERT, or is out of range; whatever
        from_pretrained raises.

    Example:
        >>> type(load_text_encoder("answerdotai/ModernBERT-base")).__name__
        'ModernBertTextEncoder'
    """
    model_type = AutoConfig.from_pretrained(checkpoint).model_type
    if model_type == _MODERNBERT_MODEL_TYPE:
        return ModernBertTextEncoder(checkpoint, trainable_top_layers)
    if trainable_top_layers:
        raise ValueError(
            f"{checkpoint} is a {model_type} model: only ModernBERT can train "
            "its top layers alone"
        )
    return PretrainedTextEncoder(checkpoint, trainable=False)
