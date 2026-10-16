"""GradientClipTally: how often each dojo's gradients hit max_grad_norm.

A step is clipped when its gradient norm, before clipping, exceeds the
phase's max_grad_norm. A dojo clipped on most of its steps trains at an
effective learning rate set by the clip, not by head_lr / encoder_lr, which
is worth seeing per round.
"""

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)


class GradientClipTally:
    """Per-dojo counts of steps taken and steps clipped, for one round."""

    def __init__(self, max_listed_dojos: int = 5) -> None:
        """
        Inputs: max_listed_dojos (int >= 1): how many of the most-clipped
            dojos the round's log line names.
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if max_listed_dojos < 1.
        """
        if max_listed_dojos < 1:
            raise ValueError("max_listed_dojos must be >= 1")
        self._max_listed_dojos = max_listed_dojos
        self._steps: Counter[str] = Counter()
        self._clipped: Counter[str] = Counter()

    def record(self, dojo_name: str, gradient_norm: float, max_norm: float) -> None:
        """Count one step of dojo_name, and whether it was clipped.

        Inputs: dojo_name; gradient_norm (the total norm before clipping, as
            torch.nn.utils.clip_grad_norm_ returns it); max_norm (the
            phase's max_grad_norm). A non-finite norm is not counted at
            all: that step is skipped, not clipped.
        Output: None.
        Side effects: updates the counts. Exceptions: none.

        Example:
            >>> tally.record("gwent_one.faction_mask", 3.2, 1.0)
        """
        if not math.isfinite(gradient_norm):
            return
        self._steps[dojo_name] += 1
        if gradient_norm > max_norm:
            self._clipped[dojo_name] += 1

    def log_and_reset(self, phase_name: str, round_index: int) -> None:
        """Log (INFO) the round's clipped share and its most-clipped dojos,
        then zero the counts. Nothing is logged for a round with no steps.

        Inputs: phase_name, round_index (for the log line).
        Output: None.
        Side effects: logs; resets the counts. Exceptions: none.

        Example:
            >>> tally.log_and_reset("joint", 3)
            # [joint] round 3: gradients clipped on 40 of 1000 step(s); most: a 12/15, b 9/30
        """
        total_steps = sum(self._steps.values())
        if total_steps:
            logger.info(
                "[%s] round %d: gradients clipped on %d of %d step(s)%s",
                phase_name,
                round_index,
                sum(self._clipped.values()),
                total_steps,
                self._most_clipped_note(),
            )
        self._steps.clear()
        self._clipped.clear()

    def _most_clipped_note(self) -> str:
        """Private helper - single caller is log_and_reset.

        Inputs: none (reads the counts).
        Output: "; most: name clipped/steps, ..." for up to
            max_listed_dojos dojos, by clipped count then clip rate; "" if
            no step was clipped.
        Side effects: none. Exceptions: none.
        """
        ranked = sorted(
            self._clipped,
            key=lambda name: (
                -self._clipped[name],
                -self._clipped[name] / self._steps[name],
                name,
            ),
        )
        listed = [
            f"{name} {self._clipped[name]}/{self._steps[name]}"
            for name in ranked[: self._max_listed_dojos]
        ]
        return f"; most: {', '.join(listed)}" if listed else ""
