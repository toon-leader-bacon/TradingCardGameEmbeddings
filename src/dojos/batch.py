from enum import Enum
from typing import List

from src.schema.type_hints import (
    BatchedTrainingInput,
    Label,
    MultiCardInput,
    MultiGroupInput,
    SingleCardInput,
    TrainingDatum,
    TrainingInput,
)


class Batch:
    class Structure_Type(Enum):
        SINGLE_CARD = SingleCardInput
        MULTI_CARDS = MultiCardInput
        MULTI_GROUP = MultiGroupInput

    def __init__(self, inputs: BatchedTrainingInput,
                 labels: List[Label]):
        self.inputs: BatchedTrainingInput = inputs
        self.labels: List[Label] = labels
        self._validate_structure()
        self.structure_type = self._determine_structure_type()

    @staticmethod
    def from_training_datum(training_datum: TrainingDatum) -> 'Batch':
        return Batch(training_datum[0], training_datum[1])

    @staticmethod
    def from_training_data(training_data: List[TrainingDatum]) -> 'Batch':
        inputs = [datum[0] for datum in training_data]
        labels = [datum[1] for datum in training_data]
        return Batch(inputs, labels)

    def _determine_structure_type(self):
        # _validate_structure() already confirmed every input shares the
        # first input's shape, so classifying the first input is enough.
        if not self.inputs:
            return self.Structure_Type.SINGLE_CARD  # Empty batch is weird but valid
        return self._structure_type_of(self.inputs[0])

    @staticmethod
    def _structure_type_of(input: TrainingInput) -> "Batch.Structure_Type":
        # TrainingInput's variants (SingleCardInput/MultiCardInput/
        # MultiGroupInput) are subscripted generics at the type-hint
        # level, but at runtime a MultiCardInput and a MultiGroupInput
        # are both plain `list` - isinstance()/issubclass() can't tell
        # them apart from the generic aliases themselves, so this
        # inspects actual element shape instead.
        if isinstance(input, SingleCardInput):
            return Batch.Structure_Type.SINGLE_CARD
        if isinstance(input, list) and all(isinstance(item, SingleCardInput) for item in input):
            return Batch.Structure_Type.MULTI_CARDS
        if isinstance(input, list) and all(isinstance(item, list) for item in input):
            return Batch.Structure_Type.MULTI_GROUP
        raise ValueError(f"Not a valid TrainingInput shape: {input}")

    def _validate_structure(self):
        if not len(self.inputs) == len(self.labels):
            raise ValueError("Inputs and labels must be the same length")
        if len(self.inputs) == 0:
            return True  # Empty batch is weird but valid

        # Ensure that all inputs are the same shape
        if not self._validate_input_structure():
            raise ValueError("Inputs are not the same type (input shape)")
        if not self._validate_label_structure():
            raise ValueError("Labels are not the same type")

    def _validate_input_structure(self):
        # Ensure the first input is a valid shape (raises otherwise).
        first_structure_type = self._structure_type_of(self.inputs[0])

        # Ensure that every input has that same shape.
        return all(
            self._structure_type_of(input) == first_structure_type for input in self.inputs
        )

    def _validate_label_structure(self):
        # Labels can be any type, so we don't need to validate the first type
        # Just ensure that all labels are of the same type
        first_label_type = type(self.labels[0])
        return all(type(label) == first_label_type for label in self.labels)
