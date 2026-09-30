"""Forward-pass settings shared by training and inference: numeric
precision (autocast) and the no-training context used whenever a model is
only scored or embedded, never trained.

Used by training/ (optimizer steps, per-round scoring) and by
CardEncoderModel.isolated_embeddings (evaluation's intrinsic embedding).
It lives here, not in training/, because running a forward pass at a given
precision is about the model; training is one of its callers.
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


class InferenceModel(ParameterOwner, Protocol):
    """A ParameterOwner with a train/eval mode (adds `training` and
    `train(mode)`): an nn.Module, or a TrainableEncoder."""

    training: bool

    def train(self, mode: bool = True) -> object: ...


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
    return parameter_device(model).type


def parameter_device(model: ParameterOwner) -> torch.device:
    """The device of the model's first parameter.

    Read on every call, so a model moved after construction is followed.

    Inputs: model (ParameterOwner).
    Output: torch.device, e.g. cuda:0; the CPU if the model has no
        parameters.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> parameter_device(model)
        device(type='cuda', index=0)
    """
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


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


@contextlib.contextmanager
def inference_context(model: InferenceModel, precision: Precision) -> Iterator[None]:
    """Run the enclosed forward passes without training: eval mode (no
    dropout), no autograd graph, and autocast at `precision`.

    Inputs: model (InferenceModel), precision (Precision).
    Output: a context manager yielding None.
    Side effects: puts model in eval mode for the duration and restores its
        prior train/eval mode on exit, even if the body raises.
    Exceptions: whatever the body raises (after the mode is restored); on
        entry, whatever autocast_for's context raises for an unsupported
        dtype.

    Example:
        >>> with inference_context(model, "fp16"):
        ...     losses = dojo.compute_loss(model(batch.inputs), batch)
    """
    was_training = model.training
    model.train(False)
    try:
        # No graph, and the requested precision, for everything in the body
        with torch.no_grad(), autocast_for(model, precision):
            yield
    finally:
        model.train(was_training)


def _autocast_dtype(precision: Precision) -> torch.dtype | None:
    """torch.float16 / torch.bfloat16 for "fp16" / "bf16"; None for "fp32"."""
    return _AUTOCAST_DTYPES.get(precision)
