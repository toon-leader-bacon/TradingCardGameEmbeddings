from typing import List, Protocol, runtime_checkable

from src.schema.type_hints import TrainingDatum


@runtime_checkable
class Mod(Protocol):
    """A transformation applied to one or more TrainingDatum before training."""

    def apply_single(self, data: TrainingDatum) -> TrainingDatum: ...

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]: ...
