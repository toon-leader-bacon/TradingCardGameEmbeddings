"""Binary cross-entropy loss for the multi-card-binary-classification
generic dojo cell (plans/dojo_v2.md).

Same shape as MseLoss (src/dojos/loss/mse_loss.py) - one raw scalar
per training example in, one ground-truth float per example - but the
scalar here is a pre-sigmoid LOGIT, and the loss is
nn.BCEWithLogitsLoss (not nn.MSELoss), expecting `labels` in [0.0, 1.0]
rather than an arbitrary regression target.

Written as a NEW file alongside the existing
src/dojos/loss/win_loss_bce_loss.py (WinLossBceLoss) rather than fixing
that file in place - human decision this session: WinLossBceLoss's own
shape (a multi-group comparison) and its unresolved +1.0/-1.0 encoding
TODO are left as-is, untouched, still unused by any dojo.
"""

from typing import List

import torch
import torch.nn as nn

from src.dojos.loss.nocab_loss import NocabLoss


class BceLoss(NocabLoss[List[torch.Tensor], List[float]]):
    """Binary cross-entropy over a single raw logit per example.

    `decoder_output` is one scalar `torch.Tensor` (a raw, un-sigmoided
    logit) per training example - the same shape MseLoss expects,
    just interpreted differently downstream.

    `labels` is the ground-truth float in {0.0, 1.0} for each
    corresponding example, same length and order as `decoder_output`.
    """

    def calculate(
        self, decoder_output: List[torch.Tensor], labels: List[float]
    ) -> torch.Tensor:
        """Binary cross-entropy between decoder_output's raw logits and
        labels' true 0.0/1.0 value.

        Inputs:
            decoder_output: this cell's decoder head output, one raw
                logit per example.
            labels: the ground-truth 0.0/1.0 label for each example,
                same length and order as decoder_output.
        Output: a scalar loss tensor.
        Side effects: none.
        Exceptions: ValueError if len(decoder_output) != len(labels).
            ValueError if any decoder_output entry isn't a
            torch.Tensor. ValueError if any label isn't a float.

        Example:
            >>> import torch
            >>> BceLoss().calculate([torch.tensor(2.0)], [1.0]).item() < 1.0
            True
        """
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not "
                f"match the number of labels ({len(labels)})"
            )
        if not all(isinstance(output, torch.Tensor) for output in decoder_output):
            raise ValueError("All decoder outputs must be torch.Tensor")
        if not all(isinstance(label, float) for label in labels):
            raise ValueError("All labels must be floats")
        func = nn.BCEWithLogitsLoss()
        target = torch.tensor(labels, dtype=torch.float32)
        return func(decoder_output, target)
