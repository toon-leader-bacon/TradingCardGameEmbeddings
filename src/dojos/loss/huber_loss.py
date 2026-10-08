"""Huber regression loss: squared error near zero, linear error far away.

This file is also a short course on why we would want it. Read the module
docstring first, then the numbers in HuberLoss, then the code.

The problem
-----------
A regression dojo predicts a number, and the loss measures how far off the
prediction `p` is from the label `y`. Call the miss `a = p - y`. We train on
z-scored labels (`LabelStats`), so `a` is measured in label standard
deviations: a = 1 means "one std off".

Three common losses, and what each one's best constant guess is:

  * MSE (mean squared error), loss = a^2. Its best guess for a set of labels
    is their MEAN. It punishes big misses quadratically.
  * MAE (mean absolute error), loss = |a|. Its best guess is the MEDIAN.
    Every miss pulls with the same force, however big.
  * Huber sits between them: it is MSE for small misses and MAE for big
    ones. Its best guess lands between the mean and the median.

Why not just MSE? Because the gradient of a^2 is 2a: a miss of 47 stds
pulls 47 times harder than a miss of 1 std, and its loss is 47^2 = 2209
times bigger. A handful of extreme labels (our heaviest tail is a label 80
stds out) can then steer a whole batch. MAE fixes that but has a constant
gradient even for a tiny miss, which makes training jittery near the
optimum. Huber takes the best of both: smooth and MSE-like where the data
is, bounded where it is not.

The formula
-----------
With the miss `a` and a threshold `delta` (both in z-scored std units):

    huber(a) = a^2                      if |a| <= delta     (the quadratic zone)
               2*delta*|a| - delta^2    if |a| >  delta     (the linear zone)

The two pieces meet at |a| = delta with the same value (delta^2) and the
same slope (2*delta), so the loss has no kink. The gradient is 2a inside the
quadratic zone and a constant +-2*delta outside it: a miss can never pull
harder than `2*delta`, however large.

Worked numbers (delta = 1.345, the default):

      a       MSE loss   Huber loss   MSE gradient   Huber gradient
     0.5          0.25         0.25           1.00             1.00
     1            1            1              2                2
     3            9            6.26           6               2.69
    10          100           25.09          20               2.69
    47         2209          124.62          94               2.69

Up to |a| = 1.345 Huber IS MSE. Past it, the 47-std miss pulls with the
force of a ~1.3-std miss instead of 47 times that of a 1-std miss. Its loss
is still large (124.6): the model is still told it is wrong, just not
catastrophically so.

Choosing delta
--------------
delta is "how big a miss before we stop trusting it as a squared error".
  * Small delta (0.1): nearly MAE, but a scaled one: with the factor 2 below
    the loss is about 2*delta*|a| = 0.2*|a| and the gradient is capped at
    2*delta = 0.2. Robust, tracks the median, and learns slowly (a small
    gradient cap) unless the learning rate or loss weight compensates.
  * Large delta (10): nearly MSE. delta -> infinity IS MSE exactly.
  * 1.345 is Huber's 1964 value. For labels whose noise is roughly Gaussian
    (std 1 after z-scoring) it keeps 95% of MSE's statistical efficiency
    while capping outliers. Because we z-score, one delta fits every dojo.
    A skewed or heavy-tailed label has an inflated std, so the bulk of its
    z-scores are smaller than 1; delta=1.345 is then relatively generous.

The cost
--------
Huber trades away "predict the mean". On a skewed label the mean and the
median differ, and Huber's answer lands between them. If a downstream use
needs the true mean (an expected value), prefer MSE for that dojo. That is
why MseLoss stays available (RegressionObjective.mse()) and why this
repository runs MSE vs. Huber as an experiment before making Huber the
default.

The factor 2
------------
torch's nn.HuberLoss uses 0.5*a^2 inside the quadratic zone, so its
gradient there is `a`. torch's nn.MSELoss omits that 0.5 (gradient 2a). Our
huber_values is exactly TWICE torch's, so inside delta it equals MSE's loss
and gradient, and delta -> infinity is exactly MseLoss. This only matters
when loss weighting is off (the raw loss trains); with weighting on, the
trainer divides the loss by its baseline, which carries the same factor, so
it cancels, except where the baseline is clamped at the weighting floor
(`baseline_floor`, 0.05): there the divisor stops tracking the factor and
the 2 is felt. That is the heavy-tail case, worth remembering when reading
a Huber dojo's gradient scale.
"""

import math
from typing import List

import torch

from src.dojos.loss.nocab_loss import NocabLoss
from src.dojos.loss.regression_target import (
    regression_target_for,
    stacked_predictions,
)

# Huber's 1964 value, in z-scored label std units
DEFAULT_HUBER_DELTA = 1.345


class HuberLoss(NocabLoss[List[torch.Tensor], List[float]]):
    """Mean Huber loss of (prediction - label), twice torch's nn.HuberLoss.

    Per example, with a = prediction - label:
        a^2                       if |a| <= delta
        2*delta*|a| - delta^2     otherwise
    Equal to MseLoss inside delta, linear beyond it (see the module
    docstring for the statistics, the worked numbers and how to choose
    delta).

    Inputs (constructor): delta (float), the zone boundary in the units of
        the labels the loss sees (z-scored std units when wrapped in
        StandardizedLabelLoss); default DEFAULT_HUBER_DELTA.
    Exceptions (constructor): ValueError if delta is not finite and > 0.

    Example: a label 47 stds out costs 124.6, not MSE's 2209, and pulls
    with a force of 2*delta = 2.69, not 94.
        >>> loss = HuberLoss()
        >>> round(loss.calculate(torch.tensor([47.0]), [0.0]).item(), 3)
        124.621
        >>> round(loss.calculate(torch.tensor([0.5]), [0.0]).item(), 3)
        0.25
    """

    def __init__(self, delta: float = DEFAULT_HUBER_DELTA) -> None:
        # Validate delta
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError(f"delta must be finite and > 0, got {delta!r}")
        self.delta = delta

    def calculate(
        self, decoder_output: List[torch.Tensor], labels: List[float]
    ) -> torch.Tensor:
        """Mean Huber loss over the batch.

        Inputs: decoder_output, labels (see regression_target_for).
        Output: scalar loss tensor.
        Side effects: none.
        Exceptions: ValueError, as regression_target_for.

        Example:
            >>> HuberLoss().calculate(torch.tensor([0.5]), [0.0]).item()
            0.25
        """
        # Validate the batch and build the target beside the predictions
        target = regression_target_for(decoder_output, labels)

        # Error per example, then the quadratic or linear zone, averaged
        predictions = stacked_predictions(decoder_output)
        return huber_values(predictions - target, self.delta).mean()


def huber_values(error: torch.Tensor, delta: float) -> torch.Tensor:
    """Per-example Huber value of a tensor of errors (2x torch's).

    The one place the formula lives: HuberLoss scores batches with it and
    best_constant_huber_loss (loss_calibration.py) scores its baseline with
    it, so a loss and its baseline cannot drift apart.

    Inputs: error (any shape), delta (> 0). Output: same shape.
    Side effects: none. Exceptions: none.

    Example:
        >>> huber_values(torch.tensor([0.5, 3.0]), delta=1.0).tolist()
        [0.25, 5.0]
    """
    magnitude = error.abs()
    quadratic = error * error
    linear = 2.0 * delta * magnitude - delta * delta
    # Both branches are finite everywhere, so where() has no NaN gradients
    return torch.where(magnitude <= delta, quadratic, linear)
