"""Per-dojo ACTIVE/SATURATED state machine driven by round normalized TEST losses.

State pattern: each dojo moves ACTIVE -> SATURATED after `patience_rounds`
rounds without a > epsilon improvement, and back to ACTIVE if its loss
rises by more than `reactivation_delta` above its loss at the moment it
saturated. "Saturated" means no further learnable signal, not "solved".

Reactivation is measured from the saturation point, not from the best
loss: a dojo that overfit before saturating (loss above its best) is not
trained while saturated, so only something else - another dojo moving a
shared encoder - can raise its loss further. Measuring from the best would
reactivate such a dojo every round, training an overfitting head more.
"""

from dataclasses import dataclass
from typing import Mapping

from src.training.plan import SaturationSpec
from src.training.recording.reports import DojoStatus


@dataclass
class _DojoTrack:
    """Mutable bookkeeping for one dojo. best_loss is None until the
    first round reports a loss; saturated_loss is the loss of the round
    that saturated the dojo (None while it never has)."""

    status: DojoStatus = DojoStatus.ACTIVE
    best_loss: float | None = None
    rounds_without_improvement: int = 0
    saturated_loss: float | None = None


class SaturationTracker:
    """Tracks the status of one phase's dojos across rounds.

    Inputs (constructor): spec (SaturationSpec), dojo_names (the phase's
        diet dojos; held-out dojos are not tracked).
    """

    def __init__(self, spec: SaturationSpec, dojo_names: tuple[str, ...]) -> None:
        self._spec = spec
        self._tracks = {name: _DojoTrack() for name in dojo_names}

    def record_round(
        self, per_dojo_normalized_loss: Mapping[str, float]
    ) -> Mapping[str, DojoStatus]:
        """Fold one round's normalized TEST losses into the tracker.

        Inputs: per_dojo_normalized_loss, the tracked dojos' normalized
            TEST loss (SplitLoss.normalized: loss / baseline, so epsilon and
            reactivation_delta are fractions of each dojo's "learned
            nothing" loss); extra names, e.g. held-out dojos, are ignored.
        Output: current status of every tracked dojo.
        Side effects: updates internal per-dojo state.
        Exceptions: none; a tracked dojo with no loss this round (its
            evaluation failed) keeps its status and counters unchanged.

        Example:
            >>> tracker.record_round({"pick": 1.2, "deck": 0.4})
        """
        result: dict[str, DojoStatus] = {}
        # Update each tracked dojo
        for name, track in self._tracks.items():
            if name not in per_dojo_normalized_loss:
                result[name] = track.status
                continue
            loss = per_dojo_normalized_loss[name]
            if track.status is DojoStatus.ACTIVE:
                self._advance_active(track, loss)
            else:
                self._consider_reactivation(track, loss)
            result[name] = track.status
        return result

    def active_dojo_names(self) -> list[str]:
        """Names of tracked dojos currently ACTIVE (all, before round 1).

        Inputs: none. Output: list[str]. Side effects: none. Exceptions: none.

        Example:
            >>> tracker.active_dojo_names()
            ['pick', 'deck']
        """
        return [
            name
            for name, track in self._tracks.items()
            if track.status is DojoStatus.ACTIVE
        ]

    def phase_done(self, quarantined: frozenset[str] = frozenset()) -> bool:
        """True once the SATURATED fraction reaches target_saturated_fraction.

        Inputs: quarantined (dojos out of the diet for failing; they count
            as saturated so they cannot block the phase from finishing).
        Output: bool. Side effects: none. Exceptions: none.

        Example:
            >>> tracker.phase_done()
            False
        """
        active = set(self.active_dojo_names()) - quarantined
        saturated = len(self._tracks) - len(active)
        return saturated / len(self._tracks) >= self._spec.target_saturated_fraction

    def _advance_active(self, track: _DojoTrack, loss: float) -> None:
        """Count improvement or stagnation; saturate after patience_rounds.

        Improvement is measured against the best loss recorded at the last
        significant improvement (not a moving minimum), so many tiny gains
        that never exceed epsilon still count as stagnation.
        """
        if track.best_loss is None or loss < track.best_loss - self._spec.epsilon:
            track.best_loss = loss
            track.rounds_without_improvement = 0
            return
        track.rounds_without_improvement += 1
        if track.rounds_without_improvement >= self._spec.patience_rounds:
            track.status = DojoStatus.SATURATED
            track.saturated_loss = loss

    def _consider_reactivation(self, track: _DojoTrack, loss: float) -> None:
        """Return a SATURATED dojo to ACTIVE if loss > saturated_loss +
        reactivation_delta (see the module docstring for why not best).

        The new best is the regressed loss, so it must improve from there.
        """
        assert track.saturated_loss is not None  # set when SATURATED is
        if loss > track.saturated_loss + self._spec.reactivation_delta:
            track.status = DojoStatus.ACTIVE
            track.best_loss = loss
            track.rounds_without_improvement = 0
