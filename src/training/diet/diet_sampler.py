"""Which dojo trains next: the diet.

Strategy: DietSampler is the swappable policy. Proportional, Uniform and
Temperature(alpha) are one family (example_count ** alpha with alpha = 1,
0, alpha), so one sampler implements them all; the DietRule union stays in
plan.py for a readable manifest.
"""

import logging
import random
from typing import Protocol, Sequence

from src.dojos.dojo import Dojo
from src.schema.splits import Split
from src.training.plan import DietRule, Proportional, Temperature, Uniform

logger = logging.getLogger(__name__)


class DietSampler(Protocol):
    def next_dojo(self, active: Sequence[Dojo], rng: random.Random) -> Dojo:
        """Pick the dojo for the next optimizer step.

        Inputs: active (non-empty Sequence[Dojo]), rng (random.Random).
        Output: one member of active.
        Side effects: advances rng.
        Exceptions: ValueError if active is empty or has no TRAIN examples.
        """
        ...


class TemperatureDietSampler:
    """Draws a dojo with probability proportional to TRAIN example_count ** alpha.

    Inputs (constructor): alpha (float >= 0).
    """

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._counts: dict[str, int] = {}  # TRAIN counts are fixed per dojo

    def next_dojo(self, active: Sequence[Dojo], rng: random.Random) -> Dojo:
        """Draw one dojo.

        Inputs: active (non-empty Sequence[Dojo]), rng (random.Random).
        Output: one member of active.
        Side effects: advances rng.
        Exceptions: ValueError if active is empty or every dojo has zero
            TRAIN examples (a dojo whose count cannot be read counts as zero).

        Example:
            >>> TemperatureDietSampler(0.0).next_dojo([a, b], random.Random(0))
        """
        # Validate inputs
        if not active:
            raise ValueError("no active dojos to sample from")
        # Weigh each dojo by its TRAIN example count ** alpha
        counts = [self._train_count(dojo) for dojo in active]
        weights = [count**self._alpha if count > 0 else 0.0 for count in counts]
        if sum(weights) <= 0:
            raise ValueError("active dojos have no TRAIN examples")
        # Draw one
        return rng.choices(list(active), weights=weights, k=1)[0]

    def _train_count(self, dojo: Dojo) -> int:
        if dojo.name not in self._counts:
            try:
                self._counts[dojo.name] = dojo.example_count(Split.TRAIN)
            except Exception:
                # An unreadable dojo is never drawn (weight 0) rather than
                # making every draw fail; logged once since the count is cached
                logger.error(
                    "cannot count %r's TRAIN examples", dojo.name, exc_info=True
                )
                self._counts[dojo.name] = 0
        return self._counts[dojo.name]


def diet_sampler_for(rule: DietRule) -> DietSampler:
    """Factory Method: build the sampler that implements a DietRule.

    Inputs: rule (Proportional | Uniform | Temperature).
    Output: DietSampler (alpha 1, 0, or the rule's alpha respectively).
    Side effects: none.
    Exceptions: none.

    Example:
        >>> diet_sampler_for(Temperature(alpha=0.5))
    """
    if isinstance(rule, Proportional):
        return TemperatureDietSampler(alpha=1.0)
    if isinstance(rule, Uniform):
        return TemperatureDietSampler(alpha=0.0)
    if isinstance(rule, Temperature):
        return TemperatureDietSampler(alpha=rule.alpha)
    raise TypeError(f"unknown DietRule: {rule!r}")
