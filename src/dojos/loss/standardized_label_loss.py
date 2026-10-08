"""Decorator: z-score a scalar regression loss's labels before scoring.

Wraps a row-wise regression NocabLoss (MseLoss or HuberLoss) so the head learns
to predict (y - mean) / std instead of y. The batch's labels stay in label
units end to end (data constructors, mods, Batch); only the loss sees the
standardized values, so no constructor or dojo subclass changes. Built by
a regression calibration (loss_calibration.py), once the dojo's
TRAIN LabelStats are known.
"""

from typing import Any, List

import torch

from src.dojos.loss.label_stats import LabelStats
from src.dojos.loss.nocab_loss import NocabLoss


class StandardizedLabelLoss(NocabLoss[Any, List[float]]):
    """A NocabLoss that standardizes float labels, then delegates.

    Inputs (constructor): inner (NocabLoss[Any, List[float]]), the loss in
        standardized units (e.g. MseLoss() or HuberLoss()); label_stats (LabelStats), the
        dojo's TRAIN stats.
    """

    def __init__(self, inner: NocabLoss[Any, List[float]], label_stats: LabelStats):
        """Side effects: none. Exceptions: none."""
        self.inner = inner
        self.label_stats = label_stats

    def calculate(self, decoder_output: Any, labels: List[float]) -> torch.Tensor:
        """inner's loss against labels standardized with label_stats.

        Inputs: decoder_output (the regression head's output, in
            standardized units), labels (List[float], label units).
        Output: scalar loss tensor (for MseLoss: MSE in std units, so 1.0
            is the mean predictor's TRAIN loss; for HuberLoss: Huber loss
            in std units, whose baseline is the best constant's loss).
        Side effects: none.
        Exceptions: whatever inner.calculate raises (length mismatch,
            non-float label).

        Example:
            >>> loss = StandardizedLabelLoss(MseLoss(), LabelStats(2.0, 1.0))
            >>> loss.calculate(torch.tensor([0.0]), [2.0]).item()
            0.0
        """
        result: torch.Tensor
        # Labels arrive in label units; the head predicts in std units
        standardized = self.label_stats.standardize(labels)
        result = self.inner.calculate(decoder_output, standardized)
        return result
