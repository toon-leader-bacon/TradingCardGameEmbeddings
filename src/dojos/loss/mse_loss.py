from typing import List

import torch
import torch.nn as nn

from src.dojos.loss.nocab_loss import NocabLoss


class MseLoss(NocabLoss[List[torch.Tensor], List[float]]):
    """Mean-squared-error regression loss for a scalar-per-example prediction.

    `decoder_output` is one scalar `torch.Tensor` per training example - the
    same shape whether that example was a single card or a pooled multi-card
    input (e.g. an estimated win rate for a card or for a whole deck).
    We expect the decoder head to have already reduced either case to one scalar.

    `labels` is the ground-truth float for each corresponding example, same
    length and order as `decoder_output`.
    """

    def calculate(
        self, decoder_output: List[torch.Tensor], labels: List[float]
    ) -> torch.Tensor:
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not "
                f"match the number of labels ({len(labels)})"
            )
        if not all(isinstance(output, torch.Tensor) for output in decoder_output):
            raise ValueError("All decoder outputs must be torch.Tensor")
        if not all(isinstance(label, float) for label in labels):
            raise ValueError("All labels must be floats")
        func = nn.MSELoss()
        target = torch.tensor(labels, dtype=torch.float32)
        return func(decoder_output, target)
