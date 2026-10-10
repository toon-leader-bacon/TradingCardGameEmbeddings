"""Weighing dojos by TRAIN example count: shared by the flat-rule sampler
(diet_sampler.py) and the table sampler (table_diet.py)."""

import logging

from src.dojos.dojo import Dojo
from src.schema.splits import Split
from src.training.plan import FlatDietRule, Proportional, Temperature, Uniform

logger = logging.getLogger(__name__)


class TrainCountCache:
    """Each dojo's TRAIN example count, read once (counts are fixed per
    dojo). A dojo whose count cannot be read counts as zero, so it is never
    drawn rather than making every draw fail; logged once."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def count(self, dojo: Dojo) -> int:
        """dojo's TRAIN example count.

        Inputs: dojo. Output: int (0 if unreadable).
        Side effects: reads the count on first call per dojo; logs an
            unreadable one at ERROR.
        Exceptions: none.

        Example:
            >>> TrainCountCache().count(dojo)
            48213
        """
        if dojo.name not in self._counts:
            try:
                self._counts[dojo.name] = dojo.example_count(Split.TRAIN)
            except Exception:
                logger.error(
                    "cannot count %r's TRAIN examples", dojo.name, exc_info=True
                )
                self._counts[dojo.name] = 0
        return self._counts[dojo.name]


def alpha_of(rule: FlatDietRule) -> float:
    """The exponent a flat rule weighs TRAIN counts by: 1 for
    Proportional, 0 for Uniform, the rule's own for Temperature.

    Inputs: rule. Output: float >= 0. Side effects: none.
    Exceptions: TypeError for anything else.

    Example:
        >>> alpha_of(Temperature(alpha=0.3))
        0.3
    """
    if isinstance(rule, Proportional):
        return 1.0
    if isinstance(rule, Uniform):
        return 0.0
    if isinstance(rule, Temperature):
        return rule.alpha
    raise TypeError(f"not a flat DietRule: {rule!r}")


def count_weight(count: int, alpha: float) -> float:
    """count ** alpha, or 0 for a dojo with no TRAIN examples (never drawn,
    even under Uniform).

    Inputs: count (>= 0), alpha (>= 0). Output: float. Side effects: none.
    Exceptions: none.

    Example:
        >>> count_weight(100, 0.5)
        10.0
    """
    return count**alpha if count > 0 else 0.0
