from typing import List

import torch
import torch.nn.functional as F

from src.dojos.loss.nocab_loss import NocabLoss


class PickPredictionCrossEntropyLoss(NocabLoss[List[torch.Tensor], List[int]]):
    """Cross-entropy loss for a pick-prediction decoder head.

    `decoder_output` is one 1-D logits tensor per training datum - one logit
    per candidate card in that datum's pack. Ragged: pack size shrinks over
    the course of a draft, so each tensor in the list may have a different
    length and the list can't be collapsed into one rectangular tensor.

    `labels` is the index of the actually-picked card within that datum's
    pack (matching decoder_output's per-datum logits), same length and
    order as decoder_output.

    Since decoder_output is ragged, each datum's cross-entropy is computed
    on its own (F.cross_entropy expects one fixed number of classes per
    call); the per-datum losses - each a single scalar, so no longer
    ragged - are then stacked and averaged into the one scalar loss every
    NocabLoss.calculate is expected to return.
    """

    def calculate(self,
                  decoder_output: List[torch.Tensor],
                  labels: List[int]) -> torch.Tensor:
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not match the number of labels ({len(labels)})")

        per_datum_losses = [
            F.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
            for logits, label in zip(decoder_output, labels)
        ]
        return torch.stack(per_datum_losses).mean()
