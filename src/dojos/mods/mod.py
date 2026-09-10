from abc import ABC, abstractmethod
from typing import List

from src.schema.type_hints import TrainingDatum


class Mod(ABC):
    """A transformation applied to one or more TrainingDatum before training.

    train_only decides which splits this mod runs against. Defaults to
    True - the original use case (noise/augmentation a model shouldn't
    see at eval time: shuffling a deck, reordering json keys, etc).
    Pass train_only=False at construction for a mod that's a structural
    requirement of the task itself rather than augmentation - e.g. a
    masking mod, where the masked field must stay masked at test/
    validation time too, or the model can just read the answer off the
    input.
    """

    def __init__(self, train_only: bool = True) -> None:
        self.train_only = train_only

    @abstractmethod
    def apply_single(self, data: TrainingDatum) -> TrainingDatum: ...

    @abstractmethod
    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]: ...
