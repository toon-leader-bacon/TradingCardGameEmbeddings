from typing import Protocol, Sequence, TypeVar, runtime_checkable

import torch

DecoderOutputT = TypeVar("DecoderOutputT", contravariant=True)
LabelsT = TypeVar("LabelsT", contravariant=True)


@runtime_checkable
class NocabLoss(Protocol[DecoderOutputT, LabelsT]):
    """A dojo-private loss: decoder output + labels -> one scalar loss tensor."""

    def calculate(
        self, decoder_output: DecoderOutputT, labels: LabelsT
    ) -> torch.Tensor: ...


def device_of(decoder_output: Sequence[torch.Tensor]) -> torch.device:
    """The device decoder_output lives on, so a loss can build its targets
    beside it rather than on the CPU.

    Inputs: decoder_output, either a real stacked Tensor (what most decoder
        heads return at runtime despite their List[torch.Tensor] hint) or a
        list of per-example Tensors.
    Output: its device; the CPU for an empty list.
    Side effects: none. Exceptions: none.

    Example:
        >>> device_of([torch.zeros(2)])
        device(type='cpu')
    """
    if isinstance(decoder_output, torch.Tensor):
        return decoder_output.device
    return decoder_output[0].device if decoder_output else torch.device("cpu")
