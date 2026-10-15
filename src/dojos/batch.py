from typing import Generic, List, cast

from src.schema.type_hints import (
    BatchedTrainingInput,
    InputShape,
    LabelT,
    TrainingDatum,
    input_shape_of,
)


class Batch(Generic[LabelT]):
    """Examples of one input shape, with labels of the dojo's label type
    (LabelT, see src/schema/type_hints.py)."""

    def __init__(self, inputs: BatchedTrainingInput, labels: List[LabelT]):
        self.inputs: BatchedTrainingInput = inputs
        self.labels: List[LabelT] = labels
        self._validate_structure()
        self.structure_type = self._determine_structure_type()

    def __len__(self) -> int:
        """Example count (not card count)."""
        return len(self.inputs)

    @staticmethod
    def from_training_datum(
        training_datum: TrainingDatum[LabelT],
    ) -> "Batch[LabelT]":
        # A one-element batch; see from_training_data's cast for why mypy
        # needs the nudge.
        return Batch(
            cast(BatchedTrainingInput, [training_datum[0]]), [training_datum[1]]
        )

    @staticmethod
    def from_training_data(
        training_data: List[TrainingDatum[LabelT]],
    ) -> "Batch[LabelT]":
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
        # LabelT is erased at runtime, so check the labels share one type
        first_label_type = type(self.labels[0])
        return all(type(label) is first_label_type for label in self.labels)
