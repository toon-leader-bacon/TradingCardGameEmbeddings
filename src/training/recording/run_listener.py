"""Observer hooks: logging (and, later, evaluation/) attach here."""

import logging
from typing import Protocol

from src.training.recording.reports import CheckpointRecord, DojoStatus, RoundReport

logger = logging.getLogger(__name__)


class RunListener(Protocol):
    def on_round_end(self, report: RoundReport) -> None:
        """Called after each round's evaluation. Side effects: listener-defined
        (must not mutate the model). Exceptions: the Trainer logs and skips
        them, so a listener bug cannot stop a run."""
        ...

    def on_checkpoint(self, record: CheckpointRecord) -> None:
        """Called after a checkpoint is written. Same contract as on_round_end."""
        ...


class LoggingRunListener:
    """Logs each round's per-dojo TEST losses and each checkpoint path."""

    def on_round_end(self, report: RoundReport) -> None:
        """Log per-dojo losses. Input: RoundReport. Output: None. Side
        effects: emits log lines. Exceptions: none."""
        losses = ", ".join(
            f"{name}={loss:.4f}" for name, loss in report.per_dojo_test_loss.items()
        )
        saturated = sorted(
            name
            for name, status in report.statuses.items()
            if status is DojoStatus.SATURATED
        )
        logger.info(
            "[%s] round %d (step %d) test loss: %s | saturated: %s | quarantined: %s",
            report.phase,
            report.round_index,
            report.step,
            losses,
            saturated,
            sorted(report.quarantined),
        )

    def on_checkpoint(self, record: CheckpointRecord) -> None:
        """Log the checkpoint path. Input: CheckpointRecord. Output: None.
        Side effects: emits a log line. Exceptions: none."""
        logger.info("checkpoint written: %s", record.path)
