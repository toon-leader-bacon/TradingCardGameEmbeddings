"""The input checks and target tensor every scalar-regression loss shares."""

from typing import List

import torch

from src.dojos.loss.nocab_loss import device_of


def stacked_predictions(decoder_output: List[torch.Tensor]) -> torch.Tensor:
    """decoder_output as one tensor: a stacked Tensor (what heads return at
    runtime) is returned as is, a list of per-example scalars is stacked.

    Inputs: decoder_output. Output: tensor of shape (N,).
    Side effects: none. Exceptions: whatever torch.stack raises for an
        empty or ragged list.

    Example:
        >>> stacked_predictions([torch.tensor(1.0), torch.tensor(2.0)])
        tensor([1., 2.])
    """
    if isinstance(decoder_output, torch.Tensor):
        return decoder_output
    return torch.stack(decoder_output)


def regression_target_for(
    decoder_output: List[torch.Tensor], labels: List[float]
) -> torch.Tensor:
    """Validate a regression batch and build its target tensor.

    Inputs: decoder_output (one scalar prediction per example; at runtime a
        stacked Tensor, whatever the type hint says), labels (List[float],
        same length and order).
    Output: float32 tensor of the labels on decoder_output's device, with
        shape (len(labels),). A stacked decoder_output must have exactly
        that shape, so a loss never silently broadcasts an (N,) target
        against an (N, 1) output.
    Side effects: none.
    Exceptions: ValueError if the lengths differ, an output is not a
        torch.Tensor, a label is not a float, or a stacked output's shape
        is not (len(labels),).

    Example:
        >>> regression_target_for(torch.zeros(2), [1.0, 2.0])
        tensor([1., 2.])
    """
    # Validate lengths, element types and label types
    if len(decoder_output) != len(labels):
        raise ValueError(
            f"The number of decoder outputs ({len(decoder_output)}) does not "
            f"match the number of labels ({len(labels)})"
        )
    if not all(isinstance(output, torch.Tensor) for output in decoder_output):
        raise ValueError("All decoder outputs must be torch.Tensor")
    if not all(isinstance(label, float) for label in labels):
        raise ValueError("All labels must be floats")
    if isinstance(decoder_output, torch.Tensor) and decoder_output.shape != (
        len(labels),
    ):
        raise ValueError(
            f"decoder output has shape {tuple(decoder_output.shape)}, "
            f"want ({len(labels)},)"
        )

    # Build the target beside the predictions
    return torch.tensor(labels, dtype=torch.float32, device=device_of(decoder_output))
