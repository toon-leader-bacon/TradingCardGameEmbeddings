"""Mean-squared-error Loss, for scalar-regression-shaped dojos.

See loss.py for the Loss contract this implements.
"""

import torch

from src.dojos.losses.loss import Loss


class MSELoss(Loss[torch.Tensor, float]):
    """Standard mean-squared error over scalar predictions.

    predictions passed to compute() are expected to be one scalar per
    example (shape (batch_size,)); labels are expected to be plain
    floats. No constructor parameters — unlike CrossEntropyLoss's
    label_smoothing, MSE has no comparable knob worth exposing yet;
    add one if a real need for it appears.
    """

    def __init__(self) -> None:
        """Construct a mean-squared-error loss.

        Inputs: none.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.

        Example:
            >>> loss = MSELoss()
        """
        self._criterion = torch.nn.MSELoss()

    def compute(self, predictions: torch.Tensor, labels: list[float]) -> torch.Tensor:
        """Compute mean squared error loss over a batch of scalar predictions.

        Inputs:
            predictions: one scalar prediction per example, shape
                (batch_size,).
            labels: one ground-truth float per example, same order as
                predictions.
        Output: a scalar loss tensor for this batch.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> loss = MSELoss()
            >>> loss.compute(torch.tensor([1.5, 2.0]), [1.0, 2.5])
        """
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return self._criterion(predictions, labels_tensor)
