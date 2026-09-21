"""Values the trainer emits: per-round reports, checkpoints, final result."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping


class DojoStatus(Enum):
    """Whether a dojo is still in the diet (ACTIVE) or has stopped
    yielding learnable signal (SATURATED)."""

    ACTIVE = "active"
    SATURATED = "saturated"


@dataclass(frozen=True)
class RoundReport:
    """State at the end of one round.

    per_dojo_test_loss: mean TEST loss of every registered dojo (diet or
        not, including held-out) that evaluated successfully this round;
        a dojo whose evaluation failed is omitted, not scored.
    statuses: ACTIVE/SATURATED for the current phase's dojos only.
    """

    phase: str
    round_index: int
    step: int
    per_dojo_test_loss: Mapping[str, float]
    statuses: Mapping[str, DojoStatus]
    quarantined: frozenset[str]


@dataclass(frozen=True)
class CheckpointRecord:
    """Where a checkpoint landed: full resumable state and encoder-only weights."""

    path: Path
    encoder_path: Path
    report: RoundReport


@dataclass(frozen=True)
class TrainingResult:
    """final_report: the last completed round (None if none completed).
    best_checkpoint: the best round of the last phase that wrote a checkpoint
    (scores are only comparable within a phase); None if none was written.
    stopped_early_reason: why the run ended before finishing its plan (for
    example too many consecutive failures), else None."""

    final_report: RoundReport | None
    best_checkpoint: CheckpointRecord | None
    stopped_early_reason: str | None


def mean_test_loss_over(report: RoundReport, dojo_names: tuple[str, ...]) -> float:
    """The score used to pick a phase's best checkpoint (lower is better).

    Inputs: report (RoundReport), dojo_names (the phase's diet dojos).
    Output: unweighted mean of those dojos' per-dojo TEST losses.
    Side effects: none.
    Exceptions: none; named dojos missing from the report are ignored, and
        the score is +inf if none are present (never the best).

    Example:
        >>> mean_test_loss_over(report, ("pick", "deck"))
        0.83
    """
    losses = [
        report.per_dojo_test_loss[name]
        for name in dojo_names
        if name in report.per_dojo_test_loss
    ]
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


def is_better_round(
    candidate: RoundReport, incumbent: RoundReport, dojo_names: tuple[str, ...]
) -> bool:
    """Whether candidate beats incumbent, judged on the same dojos.

    Inputs: two reports and the phase's diet dojo names.
    Output: True if candidate's mean TEST loss is strictly lower, taken over
        only the dojos scored in BOTH reports (so a round where a hard dojo
        failed to evaluate cannot win by omission); False if they share none.
    Side effects: none. Exceptions: none.

    Example:
        >>> is_better_round(new_report, best_report, ("pick", "deck"))
        True
    """
    shared = tuple(
        name
        for name in dojo_names
        if name in candidate.per_dojo_test_loss and name in incumbent.per_dojo_test_loss
    )
    if not shared:
        return False
    return mean_test_loss_over(candidate, shared) < mean_test_loss_over(
        incumbent, shared
    )
