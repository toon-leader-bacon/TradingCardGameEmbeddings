"""Which dojo trains next: the diet.

Strategy: DietSampler is the swappable policy. Proportional, Uniform and
Temperature(alpha) are one family (example_count ** alpha with alpha = 1,
0, alpha), so one sampler implements them all; TableDiet has its own
sampler (table_diet.py). Flat rules keep their own rng.choices draw rather
than going through a one-row table, so a flat-rule run's random stream (and
so its results) stays what it was before tables existed. The DietRule
union stays in plan.py for a readable manifest.
"""

import random
from typing import Protocol, Sequence

from src.dojos.dojo import Dojo
from src.training.diet.count_weighting import TrainCountCache, alpha_of, count_weight
from src.training.diet.table_diet import TableDietSampler
from src.training.plan import DietRule, TableDiet


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
        self._counts = TrainCountCache()

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
        weights = [count_weight(self._counts.count(d), self._alpha) for d in active]
        if sum(weights) <= 0:
            raise ValueError("active dojos have no TRAIN examples")
        # Draw one
        return rng.choices(list(active), weights=weights, k=1)[0]


def diet_sampler_for(rule: DietRule) -> DietSampler:
    """Factory Method: build the sampler that implements a DietRule.

    Inputs: rule (Proportional | Uniform | Temperature | TableDiet).
    Output: DietSampler: a TemperatureDietSampler at alpha_of(rule) for a
        flat rule, a TableDietSampler for a TableDiet.
    Side effects: none.
    Exceptions: TypeError for an unknown rule.

    Example:
        >>> diet_sampler_for(Temperature(alpha=0.5))
    """
    if isinstance(rule, TableDiet):
        return TableDietSampler(rule)
    return TemperatureDietSampler(alpha=alpha_of(rule))
