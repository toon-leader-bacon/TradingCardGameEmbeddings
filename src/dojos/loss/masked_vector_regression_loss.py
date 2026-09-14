"""Masked mean-squared-error loss for a fixed-width vector of
independent per-position probabilities, scored only where a valid
observation exists.

Sibling to soft_classification_loss.py: both score a
FixedClassificationDecoderHead's (batch_size, num_classes) raw-logit
output against a per-example label, but where SoftClassificationLoss
normalizes with softmax because its num_classes positions compete for
one shared probability mass, this loss squashes each position
independently with sigmoid, because these positions don't compete -
each is its own P(event | context), unrelated to the others (first
consumer: PickNumberDecayCurveDojo's take_rate_by_pick_number, see
plans/seventeen_lands_dojos.md). Also unlike SoftClassificationLoss, a
label's ABSENT key means "no valid observation for this position, skip
it entirely" - not "probability 0" - since here 0 is a real,
meaningfully different value from "unknown."
"""

from typing import Dict, List, Sequence, Tuple

import torch

from src.dojos.loss.nocab_loss import NocabLoss


class MaskedVectorRegressionLoss(NocabLoss[List[torch.Tensor], List[Dict[int, float]]]):
    """Masked, sigmoid-squashed MSE over a fixed-width output vector.

    label_values is accepted only for interface parity with every other
    loss_factory this project's generic dojo cells inject
    (Callable[[Sequence[str]], NocabLoss] - see
    SingleCardFixedClassificationDojo.__init__'s loss_factory
    docstring) - its actual string contents are never read, only its
    length. This loss has no real per-position vocabulary, just
    positional bucket indices - see PickNumberDecayCurveDojo
    (src/dojos/seventeenlands/draft_data/pick_number_decay_curve_dojo.py)
    for where that degenerate [str(i) for i in range(15)] vocabulary
    comes from.
    """

    def __init__(self, label_values: Sequence[str]) -> None:
        """
        Inputs:
            label_values: only its length is used - see class
                docstring.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._num_outputs = len(label_values)

    def calculate(
        self, decoder_output: List[torch.Tensor], labels: List[Dict[int, float]]
    ) -> torch.Tensor:
        """Masked MSE between sigmoid(decoder_output) and labels.

        Inputs:
            decoder_output: this cell's decoder head output - a real
                (batch_size, self._num_outputs) Tensor of raw logits at
                runtime (see soft_classification_loss.py's module for
                this project's List[torch.Tensor] type hint vs. runtime
                shape looseness, unchanged here).
            labels: one bucket_index -> take_rate dict per example, same
                length and order as decoder_output - as built by
                PickNumberDecayCurveDataConstructor.build()
                (src/dojos/generic/data_constructors.py). Every key must
                be in range [0, self._num_outputs), and every dict must
                be non-empty (an empty dict means the caller should have
                skipped that row entirely - see that class's own
                build() docstring).
        Output: a scalar loss tensor: each example's MSE is averaged
            only over ITS OWN observed buckets first, then averaged
            across the batch - so a card with many observed buckets
            doesn't outweigh one with only a couple.
        Side effects: none.
        Exceptions: ValueError if len(decoder_output) != len(labels).
            ValueError if any label dict is empty, or has a key outside
            [0, self._num_outputs) (the caller's bug - mirrors
            SoftClassificationLoss.calculate()'s philosophy).
        """
        result: torch.Tensor

        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not "
                f"match the number of labels ({len(labels)})"
            )

        # Scatter each label dict into a dense (num_outputs,) target +
        # mask pair, stacked into (batch, num_outputs) tensors aligned
        # to decoder_output's own positional ordering.
        targets: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for label in labels:
            target, mask = self._target_and_mask(label)
            targets.append(target)
            masks.append(mask)
        target_batch = torch.stack(targets)
        mask_batch = torch.stack(masks)

        # decoder_output is a real (batch, num_outputs) Tensor at
        # runtime despite its List[torch.Tensor] type hint - same known
        # looseness soft_classification_loss.py already documents.
        # Sigmoid each position independently (no softmax - see module
        # docstring), zero out unobserved positions via the mask,
        # average per example over only that example's observed
        # buckets, then average across the batch.
        predicted = torch.sigmoid(decoder_output)  # type: ignore[arg-type]
        squared_error = (predicted - target_batch) ** 2 * mask_batch
        per_example_loss = squared_error.sum(dim=-1) / mask_batch.sum(dim=-1)
        result = per_example_loss.mean()
        return result

    def _target_and_mask(
        self, label: Dict[int, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatter one example's bucket_index -> take_rate dict into a
        dense (target, mask) tensor pair.

        Private helper - single consumer is calculate().

        Inputs:
            label: see calculate()'s labels param - every key must be in
                range [0, self._num_outputs), and the dict must be
                non-empty.
        Output: a (target, mask) pair, each a (self._num_outputs,)
            tensor - mask[i] is 1.0 where label has key i, else 0.0;
            target[i] is label[i] where mask[i] is 1.0, else an
            arbitrary filler (0.0, never read since mask zeroes it out
            in calculate()).
        Side effects: none.
        Exceptions: ValueError if label is empty, or any key falls
            outside [0, self._num_outputs).
        """
        if not label:
            raise ValueError("label must have at least one observed bucket")

        target = torch.zeros(self._num_outputs)
        mask = torch.zeros(self._num_outputs)
        for bucket_index, take_rate in label.items():
            if not 0 <= bucket_index < self._num_outputs:
                raise ValueError(
                    f"Bucket index {bucket_index!r} is out of range "
                    f"[0, {self._num_outputs})"
                )
            target[bucket_index] = take_rate
            mask[bucket_index] = 1.0
        return target, mask
