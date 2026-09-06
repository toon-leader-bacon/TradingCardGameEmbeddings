from typing import List, cast

from src.schema.type_hints import (
    BatchedTrainingInput,
    InputShape,
    Label,
    TrainingDatum,
    input_shape_of,
)


class Batch:
    def __init__(self, inputs: BatchedTrainingInput, labels: List[Label]):
        self.inputs: BatchedTrainingInput = inputs
        self.labels: List[Label] = labels
        self._validate_structure()
        self.structure_type = self._determine_structure_type()

    @staticmethod
    def from_training_datum(training_datum: TrainingDatum) -> "Batch":
        # A one-element batch; see from_training_data's cast for why mypy
        # needs the nudge.
        return Batch(
            cast(BatchedTrainingInput, [training_datum[0]]), [training_datum[1]]
        )

    @staticmethod
    def from_training_data(training_data: List[TrainingDatum]) -> "Batch":
        inputs = [datum[0] for datum in training_data]
        labels = [datum[1] for datum in training_data]
        # _validate_structure() (run inside Batch.__init__) is what actually
        # confirms every input shares one shape; mypy can't distribute
        # List[Union[...]] into Union[List[...]] on its own.
        return Batch(cast(BatchedTrainingInput, inputs), labels)

    def _determine_structure_type(self) -> InputShape:
        # _validate_structure() already confirmed every input shares the
        # first input's shape, so classifying the first input is enough.
        if not self.inputs:
            return InputShape.SINGLE_CARD  # Empty batch is weird but valid
        return input_shape_of(self.inputs[0])

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
        first_structure_type = input_shape_of(self.inputs[0])

        # Ensure that every input has that same shape.
        return all(
            input_shape_of(input) == first_structure_type for input in self.inputs
        )

    def _validate_label_structure(self):
        # Labels can be any type, so we don't need to validate the first type
        # Just ensure that all labels are of the same type
        first_label_type = type(self.labels[0])
        return all(type(label) is first_label_type for label in self.labels)
