"""The trainer's view of an encoder model (SingleCardModel/MultiCardModel)."""

from typing import Iterator, Mapping, Protocol

from torch import Tensor, nn

from src.schema.type_hints import BatchedModelOutput, BatchedTrainingInput


class TrainableEncoder(Protocol):
    """What Trainer needs from the model, nothing more."""

    training: bool

    def __call__(self, inputs: BatchedTrainingInput) -> BatchedModelOutput: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def train(self, mode: bool = True) -> "TrainableEncoder": ...

    def state_dict(self) -> Mapping[str, Tensor]: ...

    def encoder_only_state_dict(self) -> Mapping[str, Tensor]:
        """Every weight of the model (dojo decoder heads live in their
        dojos, not the model): the published artifact."""
        ...
