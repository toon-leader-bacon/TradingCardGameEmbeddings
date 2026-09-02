from typing import List

import torch
import torch.nn as nn

from src.dojos.loss.nocab_loss import NocabLoss


class WinLossBceLoss(NocabLoss[List[torch.Tensor], List[bool]]):
    """Binary win/loss loss for a multi-group comparison (e.g. player deck vs. opponent deck).

    `decoder_output` is one scalar logit `torch.Tensor` per example, where a
    positive value predicts a win and a negative value predicts a loss.
    We expect the decoder head to have already reduced either case to one scalar,
    True == +0.0, False == +1.0.  # TODO: double check this scoring scheme

    `labels` is the ground-truth bool win/loss for each corresponding
    example, same length and order as `decoder_output`.

    TODO: double check `nn.BCEWithLogitsLoss` is the right fit here -
    it expects targets in [0, 1], not the +1.0/-1.0 encoding used below.
    """

    def calculate(self,
                  decoder_output: List[torch.Tensor],
                  labels: List[bool]) -> torch.Tensor:
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not match the number of labels ({len(labels)})")
        if not all(isinstance(output, torch.Tensor) for output in decoder_output):
            raise ValueError(f"All decoder outputs must be torch.Tensor")
        if not all(isinstance(label, bool) for label in labels):
            raise ValueError(f"All labels must be bools")
        func = nn.BCEWithLogitsLoss()
        labels_float = [1.0 if win else -1.0
                        for win in labels]
        return func(decoder_output, labels_float)
