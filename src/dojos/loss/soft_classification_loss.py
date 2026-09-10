"""Soft cross-entropy loss for a fixed-classification-shaped decoder head
scored against a probability-vector target instead of a single hard
class (plans/dojo_v2.md).

Reuses FixedClassificationDecoderHead's (batch_size, num_classes) logits
output unchanged - only the target shape and the loss formula differ
from FixedClassificationLoss. First (and currently only) consumer:
CardCharacterPredictionDojo (src/dojos/sts_gg/card_character_prediction_dojo.py),
injected via SingleCardFixedClassificationDojo's loss_factory parameter.
"""

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from src.dojos.loss.nocab_loss import NocabLoss


class SoftClassificationLoss(NocabLoss[List[torch.Tensor], List[Dict[str, float]]]):
    """Soft cross-entropy over a closed, per-metric label vocabulary.

    label_values is the same closed vocabulary FixedClassificationLoss
    takes (see its own docstring - full OBSERVED vocabulary, OTHER
    sentinel included where applicable). Unlike FixedClassificationLoss,
    each example's ground truth isn't one class but a probability
    distribution over (a subset of) label_values, so there's no single
    class index to encode to - instead each label dict is scattered
    into a dense (num_classes,) target vector aligned to label_values.

    A dict key outside label_values is the caller's bug (an incomplete
    label_values passed in, or a data problem upstream that should have
    been caught earlier), not silently absorbed - raises, mirroring
    FixedClassificationLoss._class_index()'s exact philosophy. In
    practice this is expected to never fire for CardCharacterPrediction-
    Dojo's own vocabulary (CharacterPredictionMetric.LABEL_VALUES,
    confirmed against the full run corpus) - see plans/dojo_v2.md.
    """

    def __init__(self, label_values: Sequence[str]) -> None:
        """
        Inputs:
            label_values: the closed set of characters this loss will
                ever see, in a fixed order - index i is target vector
                position i's character. Must be the metric's full
                observed vocabulary (OTHER sentinel included where
                applicable), not just its "normal" values - see this
                class's docstring.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._label_values = list(label_values)

    def calculate(
        self, decoder_output: List[torch.Tensor], labels: List[Dict[str, float]]
    ) -> torch.Tensor:
        """Soft cross-entropy between decoder_output's per-class logits
        and labels' true probability distribution.

        Inputs:
            decoder_output: this cell's decoder head output - see
                fixed_classification_loss.py's module docstring for the
                List[torch.Tensor] type hint vs. actual
                (batch_size, num_classes) runtime shape (same known
                looseness, unchanged here).
            labels: one character -> probability dict per example, same
                length and order as decoder_output. Every dict key MUST
                be a member of self._label_values; a dict's values need
                not sum to exactly 1.0 (not re-normalized here - see
                CardCharacterPredictionMetric.finalize(), which already
                guarantees this for its own output).
        Output: a scalar loss tensor, averaged over the batch.
        Side effects: none.
        Exceptions: ValueError if len(decoder_output) != len(labels).
            ValueError if any label dict has a key outside
            self._label_values (the caller passed an incomplete
            label_values - see class docstring).
        """
        result: torch.Tensor

        # Validate inputs
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not "
                f"match the number of labels ({len(labels)})"
            )

        # Scatter each label dict into a dense (num_classes,) target row,
        # stacked into one (batch_size, num_classes) target tensor aligned
        # to decoder_output's own class ordering.
        target_rows = [self._target_vector(label) for label in labels]
        target = torch.stack(target_rows)

        # One batched soft cross-entropy call over the whole
        # (batch_size, num_classes) tensor - not per-example, mirrors
        # FixedClassificationLoss's single nn.CrossEntropyLoss call shape.
        # decoder_output is a real (batch_size, num_classes) Tensor at
        # runtime despite its List[torch.Tensor] type hint - same known
        # looseness this module's docstring already documents.
        log_probs = F.log_softmax(decoder_output, dim=-1)  # type: ignore[arg-type]
        result = -(target * log_probs).sum(dim=-1).mean()
        return result

    def _target_vector(self, label: Dict[str, float]) -> torch.Tensor:
        """Scatter one example's character -> probability dict into a
        dense (num_classes,) target vector aligned to self._label_values.

        Private helper - single consumer is calculate().

        Inputs:
            label: a character -> probability dict, expected to have
                every key be a member of self._label_values.
        Output: a (len(self._label_values),) tensor, target[i] the
            probability for self._label_values[i] (0.0 for any
            character absent from label).
        Side effects: none.
        Exceptions: raises ValueError if any key in label isn't a
            member of self._label_values (via list.index()).
        """
        target = torch.zeros(len(self._label_values))
        for character, probability in label.items():
            if character not in self._label_values:
                raise ValueError(
                    f"Label character {character!r} is not a member of "
                    f"label_values {self._label_values!r}"
                )
            target[self._label_values.index(character)] = probability
        return target
