"""Baseline-normalized loss weighting: the gradient-side use of
`loss / baseline_loss`.

Dojo losses sit on unrelated scales (nats over a few classes, nats over a
batch's items, squared z-scored label units). Dividing a step's loss by
the dojo's baseline_loss puts every dojo's untrained loss near 1.0, and a
per-dojo multiplier then says how much that dojo should count: 2.0 is
twice a baseline dojo, 0.75 is three quarters of one.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.dojos.dojo import Dojo, DojoBatch

# The cap on 1 / baseline: a baseline below this is treated as this, so a
# near-deterministic dojo cannot blow up its gradient. Provisional until the
# baseline survey (plans/baseline_normalized_loss_weighting.md, step 1)
# picks it from the catalog's real baselines.
DEFAULT_BASELINE_FLOOR = 0.05


@dataclass(frozen=True)
class LossWeighting:
    """How much each dojo's step loss is scaled before backward().

    weights: dojo name -> multiplier, finite and > 0. A dojo not named
        counts 1.0. Copied into a read-only mapping on construction.
    baseline_floor: finite and > 0; a dojo baseline below it is treated as
        the floor, capping the scale at weight / baseline_floor.

    Exceptions (constructor): ValueError for a weight or floor that is not
        finite and > 0 (a weight of 0 is a mistake: take the dojo out of
        the diet instead).

    Example:
        >>> LossWeighting({"mtg.card_cmc": 0.75}, DEFAULT_BASELINE_FLOOR).weights
        mappingproxy({'mtg.card_cmc': 0.75})
    """

    weights: Mapping[str, float]
    baseline_floor: float = DEFAULT_BASELINE_FLOOR

    def __post_init__(self) -> None:
        # Validate every weight and the floor
        for name, weight in self.weights.items():
            _require_positive_finite(weight, f"loss weight for {name!r}")
        _require_positive_finite(self.baseline_floor, "baseline_floor")
        # Freeze a copy (frozen dataclass: set through object.__setattr__)
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    def scale_for(self, dojo: Dojo, batch: DojoBatch) -> float | None:
        """The factor to multiply this batch's loss by before backward().

        Inputs: dojo (Dojo), batch (a batch that dojo yielded).
        Output: weight(dojo.name) / max(dojo.baseline_loss(batch),
            baseline_floor), finite and > 0; None if the batch has no
            usable baseline (e.g. a contrastive batch with no negatives),
            which the trainer skips rather than counts as a fault.
        Side effects: none (the baseline is pure data, never the encoder).
        Exceptions: TypeError if batch is not this dojo's batch type.

        Example:
            >>> weighting = LossWeighting({"d": 2.0}, baseline_floor=0.1)
            >>> weighting.scale_for(dojo_d, batch)   # baseline 0.69
            2.898...
        """
        weight = self.weights.get(dojo.name, 1.0)
        # A batch with no usable baseline is skipped, not an error
        try:
            baseline = dojo.baseline_loss(batch)
        except ValueError:
            return None
        # Cap 1 / baseline at 1 / baseline_floor
        return weight / max(baseline, self.baseline_floor)


def _require_positive_finite(value: float, context: str) -> None:
    """Check value against LossWeighting's rule.

    Inputs: value (float), context (str naming it, for the message).
    Output: None.
    Side effects: none.
    Exceptions: ValueError naming context unless value is finite and > 0.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{context} must be finite and > 0, got {value!r}")
