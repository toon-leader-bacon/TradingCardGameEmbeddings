"""ModernBertLayerStack: run a ModernBERT model in two parts, split after
any transformer layer.

Adapter over a Hugging Face ModernBertModel (transformers 5.x): its own
forward runs embeddings -> every layer -> final norm in one call. A
token-level feature cache needs the hidden states after layer k, and
later the rest of the stack from those states. This class replays the
model's forward in two halves, with the same masks and rotary position
embeddings, so run_from(run_through(x, k), k) is the model's own output.

Positions are 0..L-1 for every text (right padding), so a text's real
tokens get the same positions however much padding a batch adds; the
masks hide padding keys. That is why states cached from one batch can be
replayed in another.
"""

import itertools
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)

# Attention implementations whose ModernBertModel.forward this class
# replays (padded batches with masks; no unpadded cu_seqlens path)
_REPLAYED_ATTENTION = frozenset({"sdpa", "eager"})


@dataclass(frozen=True)
class _StackParts:
    """The ModernBertModel submodules the split reads (checked once).

    embeddings, layers, final_norm, rotary_emb: the model's own.
    layer_types: the config's attention type per layer ("full_attention"
        or "sliding_attention"), which picks each layer's mask and rotary
        table.
    config: the model's config, which the mask builders read.
    """

    config: Any
    embeddings: nn.Module
    layers: nn.ModuleList
    final_norm: nn.Module
    rotary_emb: nn.Module
    layer_types: tuple[str, ...]


class ModernBertLayerStack:
    """A ModernBertModel, runnable through layer k and from layer k on.

    Not an nn.Module: it holds references into the model, which the text
    encoder owns (and which state_dict, .to() and train() already see).
    """

    def __init__(self, model: nn.Module) -> None:
        """Inputs: model (a transformers ModernBertModel).
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if model lacks the ModernBERT parts (an
            embeddings module, a layers list, final_norm, rotary_emb, and
            config.layer_types matching the layers).

        Example:
            >>> model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
            >>> stack = ModernBertLayerStack(model)
            >>> stack.layer_count
            22
        """
        self._model = model
        self._parts = _read_stack_parts(model)

    @property
    def layer_count(self) -> int:
        """The transformer layers in the model. Inputs: none. Output: int.
        Side effects: none. Exceptions: none."""
        return len(self._parts.layers)

    def run_through(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Token embeddings, then layers [0, layer): the hidden states a
        token-level cut stores. No final norm.

        Inputs: input_ids (N, L), attention_mask (N, L) (1 real, 0 pad),
            layer (0 = the embeddings' output, up to layer_count).
        Output: (N, L, hidden) Tensor; padded positions hold values no
            caller may read.
        Side effects: none beyond autograd.
        Exceptions: ValueError if layer is outside [0, layer_count].

        Example:
            >>> stack.run_through(ids, mask, 20).shape
            torch.Size([2, 37, 768])
        """
        _check_layer(layer, self.layer_count)
        hidden_states = self._parts.embeddings(input_ids=input_ids)
        return self._run_layers(hidden_states, attention_mask, 0, layer)

    def run_from(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Layers [layer, layer_count), then the final norm: the model's
        last_hidden_state, given run_through's output for the same layer.

        Inputs: hidden_states (N, L, hidden), attention_mask (N, L), layer
            (as run_through).
        Output: (N, L, hidden) Tensor.
        Side effects: none beyond autograd.
        Exceptions: ValueError if layer is outside [0, layer_count].

        Example:
            >>> stack.run_from(stack.run_through(ids, mask, 20), mask, 20).shape
            torch.Size([2, 37, 768])
        """
        _check_layer(layer, self.layer_count)
        result = self._run_layers(
            hidden_states, attention_mask, layer, self.layer_count
        )
        return self._parts.final_norm(result)

    def parameters_through(self, layer: int) -> Iterator[nn.Parameter]:
        """The parameters run_through(..., layer) uses: the embeddings' and
        layers [0, layer). Inputs: layer. Output: iterator.
        Side effects: none. Exceptions: ValueError for a layer out of
        range."""
        _check_layer(layer, self.layer_count)
        lower = [self._parts.embeddings, *self._parts.layers[:layer]]
        return itertools.chain.from_iterable(m.parameters() for m in lower)

    def parameters_from(self, layer: int) -> Iterator[nn.Parameter]:
        """The parameters run_from(..., layer) uses: layers [layer,
        layer_count) and the final norm. Inputs: layer. Output: iterator.
        Side effects: none. Exceptions: ValueError for a layer out of
        range."""
        _check_layer(layer, self.layer_count)
        upper = [*self._parts.layers[layer:], self._parts.final_norm]
        return itertools.chain.from_iterable(m.parameters() for m in upper)

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        first: int,
        stop: int,
    ) -> torch.Tensor:
        """Layers [first, stop) over hidden_states, each with the mask and
        rotary table its attention type needs.

        Private helper - consumers are run_through and run_from.
        Inputs: hidden_states (N, L, hidden), attention_mask (N, L), first,
            stop. Output: Tensor. Side effects: none beyond autograd.
        Exceptions: none.
        """
        if first == stop:
            return hidden_states
        parts = self._parts
        # The masks and rotary tables ModernBertModel.forward builds
        mask_kwargs = {
            "config": parts.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
        }
        masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(
                **mask_kwargs
            ),
        }
        position_ids = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        ).unsqueeze(0)
        needed = set(parts.layer_types[first:stop])
        rotary = {
            kind: parts.rotary_emb(hidden_states, position_ids, kind) for kind in needed
        }
        # Each layer reads the mask and table of its own attention type
        for index in range(first, stop):
            kind = parts.layer_types[index]
            hidden_states = parts.layers[index](
                hidden_states,
                attention_mask=masks[kind],
                position_embeddings=rotary[kind],
            )
        return hidden_states


def _read_stack_parts(model: nn.Module) -> _StackParts:
    """Parse the model into the parts the split needs.

    Private helper - single consumer is ModernBertLayerStack.__init__.
    Inputs: model. Output: _StackParts. Side effects: none.
    Exceptions: ValueError naming the first missing part.
    """
    for name in ("config", "embeddings", "layers", "final_norm", "rotary_emb"):
        if not hasattr(model, name):
            raise ValueError(
                f"{type(model).__name__} has no {name}: only ModernBERT models "
                "can be split into layers"
            )
    attention = getattr(model.config, "_attn_implementation", None)
    if attention not in _REPLAYED_ATTENTION:
        raise ValueError(
            f"attention implementation {attention!r} is not replayed by the "
            f"layer split; use one of {sorted(_REPLAYED_ATTENTION)}"
        )
    layers = model.layers
    layer_types = tuple(getattr(model.config, "layer_types", None) or ())
    if not isinstance(layers, nn.ModuleList) or len(layer_types) != len(layers):
        raise ValueError(
            f"{type(model).__name__}'s layers and config.layer_types do not "
            "match: only ModernBERT models can be split into layers"
        )
    modules = [model.embeddings, model.final_norm, model.rotary_emb]
    embeddings, final_norm, rotary_emb = [_require_module(part) for part in modules]
    return _StackParts(
        config=model.config,
        embeddings=embeddings,
        layers=layers,
        final_norm=final_norm,
        rotary_emb=rotary_emb,
        layer_types=layer_types,
    )


def _require_module(part: object) -> nn.Module:
    """Private helper - single consumer is _read_stack_parts.

    Inputs: part (a model attribute). Output: part, typed as a Module.
    Side effects: none. Exceptions: ValueError if it is not a Module.
    """
    if not isinstance(part, nn.Module):
        raise ValueError(f"expected a module, got {type(part).__name__}")
    return part


def _check_layer(layer: int, layer_count: int) -> None:
    """Private helper - consumers are every public method taking a layer.

    Inputs: layer, layer_count. Output: none. Side effects: none.
    Exceptions: ValueError if layer is outside [0, layer_count].
    """
    if not 0 <= layer <= layer_count:
        raise ValueError(f"layer must be in [0, {layer_count}], got {layer}")
