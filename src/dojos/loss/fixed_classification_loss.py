"""Cross-entropy loss for the single/multi-card-fixed-classification
generic dojo cells (plans/dojo_v2.md).

Unlike PickPredictionCrossEntropyLoss (src/dojos/loss/
pick_prediction_cross_entropy_loss.py), decoder_output here is NOT
ragged - every training datum has the same fixed num_classes
(len(label_values)), since the decoder head already stacks its output
into one rectangular (batch_size, num_classes) tensor (see
single_card_fixed_classification/decoder_head.py). So this loss does
one batched nn.CrossEntropyLoss call, not per-datum-then-stack.

decoder_output's type hint (List[torch.Tensor]) is the same known
looseness MseLoss.calculate() already carries: the decoder head
actually hands back a single stacked (batch_size, num_classes) tensor
at runtime, not a list of per-example tensors - see mse_loss.py's own
history for why that's accepted, existing debt rather than something
this class needs to fix.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from src.dojos.loss.nocab_loss import NocabLoss


class FixedClassificationLoss(NocabLoss[List[torch.Tensor], List[str]]):
    """Cross-entropy over a closed, per-metric label vocabulary.

    label_values is the metric's full OBSERVED vocabulary (see
    plans/dojo_v2.md's "label_values ... full observed vocabulary" -
    e.g. it already includes OTHER_LABEL as one of its own elements for
    a metric that falls back to it, rather than this class adding it).
    Encodes each raw string label to label_values.index(label)
    internally - a label outside label_values is the caller's bug (an
    incomplete vocabulary passed in), not a data problem to skip over,
    so it raises rather than being dropped silently.
    """

    def __init__(self, label_values: Sequence[str]) -> None:
        """
        Inputs:
            label_values: the closed set of labels this loss will ever
                see, in a fixed order - index i is class i's label.
                Must be the metric's full observed vocabulary (OTHER
                sentinel included where applicable), not just its
                "normal" values - see this class's docstring.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._label_values = list(label_values)

    def calculate(
        self, decoder_output: List[torch.Tensor], labels: List[str]
    ) -> torch.Tensor:
        """Cross-entropy between decoder_output's per-class logits and
        labels' true class.

        Inputs:
            decoder_output: this cell's decoder head output - see this
                module's docstring for the List[torch.Tensor] type hint
                vs. actual (batch_size, num_classes) runtime shape.
            labels: the ground-truth raw string label for each example,
                same length and order as decoder_output. Every label
                MUST be a member of self._label_values.
        Output: a scalar loss tensor.
        Side effects: none.
        Exceptions: ValueError if len(decoder_output) != len(labels).
            ValueError if any label isn't in self._label_values (the
            caller passed an incomplete label_values - see class
            docstring).
        """
        result: torch.Tensor

        # Validate inputs
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not "
                f"match the number of labels ({len(labels)})"
            )

        # Encode each raw string label to its fixed class index.
        # Propagates _class_index()'s ValueError for a label outside
        # self._label_values rather than catching it - see docstring.
        target_indices = [self._class_index(label) for label in labels]
        target = torch.tensor(target_indices, dtype=torch.long)

        # One batched cross-entropy call over the whole
        # (batch_size, num_classes) tensor - not ragged, see module
        # docstring.
        func = nn.CrossEntropyLoss()
        result = func(decoder_output, target)
        return result

    def _class_index(self, label: str) -> int:
        """Look up label's fixed class index in self._label_values.

        Private helper - single consumer is calculate().

        Inputs:
            label: a raw string label, expected to be a member of
                self._label_values.
        Output: label's index in self._label_values.
        Side effects: none.
        Exceptions: raises ValueError if label isn't a member of
            self._label_values (via list.index()).
        """
        return self._label_values.index(label)
