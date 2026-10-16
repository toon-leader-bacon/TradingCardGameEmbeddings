"""LearningRateSchedule: a phase's learning-rate multiplier over its steps.

A linear warmup from zero, then either a constant rate or a cosine decay
down to a floor. The default (no warmup, constant) is the trainer's
behavior before schedules existed, so a config without a schedule trains
as it always did. Before each step attempt the Trainer sets every
optimizer group's lr to its base rate (encoder_lr or head_lr) times the
multiplier.
"""

import math
from dataclasses import dataclass
from typing import Literal, get_args

DecayKind = Literal["constant", "cosine"]


@dataclass(frozen=True)
class LearningRateSchedule:
    """The multiplier on a phase's encoder_lr and head_lr at each step.

    warmup_steps: steps over which the multiplier rises linearly from
        1 / warmup_steps to 1; 0 starts at the full rate.
    decay: "constant" holds 1 after the warmup; "cosine" falls from 1 to
        min_lr_fraction along half a cosine, reaching it at the phase's
        last possible step (Phase.max_steps).
    min_lr_fraction: the cosine's floor, in [0, 1]; unused by "constant".
    """

    warmup_steps: int = 0
    decay: DecayKind = "constant"
    min_lr_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.decay not in get_args(DecayKind):
            raise ValueError(
                f"decay must be one of {get_args(DecayKind)}, got {self.decay!r}"
            )
        # `not` also rejects NaN
        if not 0 <= self.min_lr_fraction <= 1:
            raise ValueError("min_lr_fraction must be in [0, 1]")

    def multiplier(self, step: int, total_steps: int) -> float:
        """The learning-rate multiplier for a 0-based step of a phase.

        Inputs: step (int >= 0, the phase's steps so far); total_steps
            (int >= 1, the phase's step budget). A step past the budget
            keeps the final multiplier.
        Output: float in [0, 1]: (0, 1] during the warmup (which may sit
            below min_lr_fraction), then [min_lr_fraction, 1] (0 only for
            a cosine with a 0 floor, at its last step).
        Side effects: none.
        Exceptions: none.

        Example:
            >>> LearningRateSchedule(warmup_steps=10).multiplier(4, 100)
            0.5
            >>> LearningRateSchedule(decay="cosine").multiplier(50, 101)
            0.5
        """
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.decay == "constant":
            return 1.0
        # Cosine from the end of the warmup to the last step of the budget
        decay_steps = max(1, total_steps - 1 - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_fraction + (1.0 - self.min_lr_fraction) * cosine
