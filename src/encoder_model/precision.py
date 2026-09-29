"""Numeric precision for a model's forward pass: one home for autocast.

Used by training/ (optimizer steps, per-round scoring) and by the
CardEmbedder adapters (evaluation's intrinsic embedding). It lives here,
not in training/, because running a forward pass at a given precision is
about the model; training is one of its callers.
"""

import contextlib
from typing import Any, ContextManager, Iterator, Literal, Protocol

import torch
from torch import nn

Precision = Literal["fp32", "fp16", "bf16"]

_AUTOCAST_DTYPES: dict[Precision, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class ParameterOwner(Protocol):
    """Anything exposing its parameters: an nn.Module, or a
    TrainableEncoder (which is not typed as an nn.Module)."""

    def parameters(self) -> Iterator[nn.Parameter]: ...


def device_type_of(model: ParameterOwner) -> str:
    """The device type of the model's first parameter.

    Read on every call, so a model moved after construction is followed.

    Inputs: model (ParameterOwner).
    Output: str, e.g. "cpu", "cuda" ("cuda" also on ROCm), "mps"; "cpu" if
        the model has no parameters.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> device_type_of(model)
        'cuda'
    """
    for parameter in model.parameters():
        return parameter.device.type
    return "cpu"


def autocast_for(model: ParameterOwner, precision: Precision) -> ContextManager[Any]:
    """A context that runs forward passes at `precision` on the model's device.

    Weights stay float32; only the ops inside the context run in the lower
    precision.

    Inputs: model (ParameterOwner), precision (Precision).
    Output: torch.autocast(device_type_of(model), dtype) for "fp16"/"bf16";
        contextlib.nullcontext() for "fp32".
    Side effects: none until entered.
    Exceptions: none here; torch may raise on entry if the device does not
        support the dtype.

    Example:
        >>> with autocast_for(model, "fp16"):
        ...     embeddings = model(cards)
    """
    dtype = _autocast_dtype(precision)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type_of(model), dtype=dtype)


def _autocast_dtype(precision: Precision) -> torch.dtype | None:
    """torch.float16 / torch.bfloat16 for "fp16" / "bf16"; None for "fp32"."""
    return _AUTOCAST_DTYPES.get(precision)
