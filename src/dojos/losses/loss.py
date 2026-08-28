"""The shared, reusable loss-computation contract dojos may compose from.

See plans/training_pipeline.md's note on shared loss injection. A Loss
is a plain Strategy object (typing.Protocol) a dojo receives via its
own constructor and delegates to internally from compute_loss —
training/ never constructs, sees, or calls one. This keeps loss-
hyperparameter tuning in one place (whatever constructs a dojo) and
lets dojos of similar shape (e.g. every classification-style dojo)
reuse the same implementation instead of hand-rolling loss math per
dojo.
"""

from typing import Protocol, TypeVar

import torch

PredictionsT = TypeVar("PredictionsT", contravariant=True)
LabelsT = TypeVar("LabelsT")  # invariant: wrapped in list[LabelsT] below, list is invariant


class Loss(Protocol[PredictionsT, LabelsT]):
    """One reusable loss computation, injected into a dojo at construction."""

    def compute(self, predictions: PredictionsT, labels: list[LabelsT]) -> torch.Tensor:
        """Compute a scalar loss from a decoder head's predictions and ground truth.

        Inputs:
            predictions: this batch's decoder head output — shape is
                whichever concrete Loss/dojo pairing agrees on (e.g.
                per-class logits for a classification Loss).
            labels: this batch's ground truth, same order as
                predictions.
        Output: a scalar loss tensor for this batch.
        Side effects: none.
        Exceptions: none.
        """
        ...
