from typing import List

import torch
import torch.nn.functional as F

from src.dojos.loss.nocab_loss import NocabLoss


class PickPredictionCrossEntropyLoss(NocabLoss[torch.Tensor, List[int]]):
    """Cross-entropy loss for a pick-prediction decoder head.

    `decoder_output` is one row of logits per training datum, one logit
    per candidate card in that datum's pack, padded to the batch's
    largest pack with -inf (pack size shrinks over the course of a draft).
    A -inf logit gets probability 0, so padding never competes with a
    real option, and one F.cross_entropy call covers the whole batch.

    `labels` is the index of the actually-picked card within that datum's
    pack (matching decoder_output's row), same length and order as
    decoder_output's rows.
    """

    def calculate(
        self, decoder_output: torch.Tensor, labels: List[int]
    ) -> torch.Tensor:
        """Mean cross-entropy of the picked option, over the batch.

        Inputs: decoder_output, (examples, max_options) logits, -inf at
            padding; labels, one option index per example, each pointing
            at a real (finite) logit.
        Output: scalar tensor, the per-example mean.
        Side effects: one host-to-device copy of labels.
        Exceptions: ValueError if len(labels) != decoder_output's row
            count, or a label is outside [0, max_options).

        Example:
            >>> PickPredictionCrossEntropyLoss().calculate(logits, [0, 2])
            tensor(1.0986)
        """
        # Validate inputs on the host: no device sync
        _check_labels(decoder_output, labels)

        # One call for the whole batch
        targets = torch.tensor(labels, device=decoder_output.device)
        return F.cross_entropy(decoder_output, targets)


def _check_labels(decoder_output: torch.Tensor, labels: List[int]) -> None:
    """Private helper - single caller is calculate.

    Inputs: decoder_output, labels.
    Output: none. Side effects: none (reads only decoder_output's shape).
    Exceptions: ValueError on a row-count mismatch or an out-of-range
        label. A label on a padding slot gives an infinite loss, which
        the trainer's non-finite check reports; checking it here would
        cost a device sync.
    """
    example_count, max_options = decoder_output.shape
    if len(labels) != example_count:
        raise ValueError(
            f"The number of decoder outputs ({example_count}) does not "
            f"match the number of labels ({len(labels)})"
        )
    for label in labels:
        if not 0 <= label < max_options:
            raise ValueError(f"label {label} outside the {max_options} options")
