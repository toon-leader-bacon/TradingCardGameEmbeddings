from typing import Protocol, TypeVar, runtime_checkable

import torch

DecoderOutputT = TypeVar("DecoderOutputT", contravariant=True)
LabelsT = TypeVar("LabelsT", contravariant=True)


@runtime_checkable
class NocabLoss(Protocol[DecoderOutputT, LabelsT]):
    """A dojo-private loss: decoder output + labels -> one scalar loss tensor."""

    def calculate(
        self, decoder_output: DecoderOutputT, labels: LabelsT
    ) -> torch.Tensor: ...
