"""Cross-entropy Loss, for classification-shaped dojos.

See loss.py for the Loss contract this implements.
"""

import torch

from src.dojos.losses.loss import Loss


class CrossEntropyLoss(Loss[torch.Tensor, int]):
    """Standard cross-entropy over per-class logits.

    predictions passed to compute() are expected to be per-class
    logits (shape (batch_size, num_classes)); labels are expected to
    be integer class indices.
    """

    def __init__(self, label_smoothing: float) -> None:
        """Construct a cross-entropy loss with the given label smoothing.

        Inputs:
            label_smoothing: passed straight through to
                torch.nn.CrossEntropyLoss's own label_smoothing.
        Output: none (constructor).
        Side effects: none.
        Exceptions: raises ValueError if label_smoothing is not in
            [0, 1).

        Example:
            >>> loss = CrossEntropyLoss(label_smoothing=0.1)
        """
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self._criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def compute(self, predictions: torch.Tensor, labels: list[int]) -> torch.Tensor:
        """Compute mean cross-entropy loss over a batch of logits.

        Inputs:
            predictions: per-class logits, shape (batch_size, num_classes).
            labels: integer class indices, one per row of predictions,
                same order.
        Output: a scalar loss tensor for this batch.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> loss = CrossEntropyLoss(label_smoothing=0.0)
            >>> loss.compute(torch.randn(4, 3), [0, 1, 2, 0])
        """
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return self._criterion(predictions, labels_tensor)
