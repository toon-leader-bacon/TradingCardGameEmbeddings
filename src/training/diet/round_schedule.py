"""RoundSchedule: which dojo each step of a round trains on, drawn once
when the round starts.

The diet samplers depend only on (active dojos, rng), so drawing a
round's steps up front gives the draws the trainer would have made step
by step. A BatchFeed builds batches in schedule order, so a prefetch
worker knows exactly which batches the trainer will ask for.
"""

import random
from dataclasses import dataclass
from typing import Sequence

from src.dojos.dojo import Dojo
from src.training.diet.batch_feed_errors import ScheduleMismatchError
from src.training.diet.diet_sampler import DietSampler


@dataclass(frozen=True)
class RoundSchedule:
    """The dojo name of each step, in order; never empty.

    dojo_names: one per step.
    """

    dojo_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Inputs: none (checks self). Output: none. Side effects: none.
        Exceptions: ValueError if dojo_names is empty."""
        if not self.dojo_names:
            raise ValueError("a round schedule needs at least one step")


def draw_round_schedule(
    sampler: DietSampler, active: Sequence[Dojo], steps: int, rng: random.Random
) -> RoundSchedule:
    """steps draws of sampler.next_dojo(active, rng), in order.

    Inputs: sampler (the phase's diet), active (non-empty), steps (>= 1),
        rng (the trainer's).
    Output: RoundSchedule of steps dojo names.
    Side effects: advances rng (and the sampler's caches) exactly as
        steps step-by-step draws would.
    Exceptions: ValueError if active is empty or steps < 1; whatever the
        sampler raises (ValueError when no active dojo has TRAIN examples).

    Example:
        >>> draw_round_schedule(sampler, [a, b], 4, random.Random(0)).dojo_names
        ('a', 'b', 'a', 'a')
    """
    # Validate inputs
    if not active:
        raise ValueError("a round needs at least one active dojo")
    if steps < 1:
        raise ValueError(f"a round needs at least one step, got {steps}")

    # One draw per step, as the trainer once drew them
    names = [sampler.next_dojo(active, rng).name for _ in range(steps)]
    return RoundSchedule(tuple(names))


def require_next_step(pending: Sequence[str] | None, name: str) -> None:
    """The check every BatchFeed.next_batch makes before taking a step.

    Inputs: pending (the round's unconsumed dojo names, next first; None
        when no round is open), name (the dojo the trainer asks for).
    Output: none. Side effects: none.
    Exceptions: ScheduleMismatchError if no round is open, the schedule is
        used up, or name is not the next scheduled dojo.

    Example:
        >>> require_next_step(["a", "b"], "a")
    """
    if pending is None:
        raise ScheduleMismatchError(f"next_batch({name!r}) with no round open")
    if not pending:
        raise ScheduleMismatchError(f"next_batch({name!r}) past the round's schedule")
    if pending[0] != name:
        raise ScheduleMismatchError(
            f"next_batch({name!r}), but the next scheduled dojo is {pending[0]!r}"
        )
