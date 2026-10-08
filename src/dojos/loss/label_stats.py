"""TRAIN-split statistics of a scalar regression label, for z-scoring.

A regression dojo trains on (y - mean) / std so its loss (MSE or Huber)
is in "label standard deviations" rather than raw units. The stats come from the
TRAIN split only (test labels would leak otherwise) and live on the dojo
(`GenericDojo.label_stats`), so a prediction can be mapped back to label
units for a report.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class LabelStats:
    """Mean and population standard deviation of a dojo's TRAIN labels.

    mean: TRAIN label mean, in label units.
    std: TRAIN label population standard deviation (ddof=0), in label
        units; always finite and > 0 (see from_labels). Population, not
        sample, std so the mean predictor's standardized TRAIN MSE is
        exactly 1.0 - the regression baseline.
    """

    mean: float
    std: float

    @classmethod
    def from_labels(cls, labels: Sequence[float]) -> "LabelStats":
        """Parse a TRAIN label sample into LabelStats (Factory Method).

        Inputs: labels (Sequence[float]), the TRAIN split's labels (or a
            sample of them), in label units.
        Output: LabelStats of those labels.
        Side effects: none.
        Exceptions: ValueError if labels is empty, holds a non-finite
            value, or has zero standard deviation (a constant label is a
            construction error, not a silent divide - the plan's rule).

        Example:
            >>> LabelStats.from_labels([1.0, 3.0])
            LabelStats(mean=2.0, std=1.0)
        """
        result: LabelStats

        # Validate inputs: non-empty and finite
        _require_finite_labels(labels)

        # Population mean and std in one pass over the sample
        mean, std = _mean_and_population_std(labels)
        # Compare the labels, not std: rounding can leave a constant
        # sample with a std of ~1e-17 instead of exactly 0
        if std == 0.0 or min(labels) == max(labels):
            raise ValueError(
                f"TRAIN labels have zero standard deviation (all {mean!r}); "
                "a constant regression label cannot be z-scored"
            )
        result = cls(mean=mean, std=std)
        return result

    def standardize(self, labels: Sequence[float]) -> List[float]:
        """labels in standard-deviation units: (label - mean) / std.

        Inputs: labels (Sequence[float]) in label units.
        Output: List[float], same length and order.
        Side effects: none. Exceptions: none.

        Example:
            >>> LabelStats(2.0, 1.0).standardize([1.0, 3.0])
            [-1.0, 1.0]
        """
        return [(label - self.mean) / self.std for label in labels]

    def to_label_units(self, standardized_value: float) -> float:
        """Map one standardized prediction back to label units:
        value * std + mean.

        Inputs: standardized_value (float), e.g. a regression head's output.
        Output: float in label units.
        Side effects: none. Exceptions: none.

        Example:
            >>> LabelStats(2.0, 1.0).to_label_units(-1.0)
            1.0
        """
        return standardized_value * self.std + self.mean


def _require_finite_labels(labels: Sequence[float]) -> None:
    """Raise ValueError if labels is empty or holds a NaN/inf.

    Inputs: labels (Sequence[float]). Output: None.
    Side effects: none. Exceptions: ValueError as above.
    """
    if not labels:
        raise ValueError("cannot compute label stats from no labels")
    for label in labels:
        if not math.isfinite(label):
            raise ValueError(f"TRAIN labels must be finite, got {label!r}")


def _mean_and_population_std(labels: Sequence[float]) -> tuple[float, float]:
    """(mean, population std) of a non-empty, finite label sequence.

    Inputs: labels (Sequence[float]), already validated.
    Output: (mean, std) floats; std may be 0.0 (the caller decides).
    Side effects: none. Exceptions: none.
    """
    mean = math.fsum(labels) / len(labels)
    variance = math.fsum((label - mean) ** 2 for label in labels) / len(labels)
    return mean, math.sqrt(variance)
