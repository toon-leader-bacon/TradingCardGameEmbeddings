"""Observer hooks: logging (and, later, evaluation/) attach here."""

import csv
import logging
from pathlib import Path
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


class CsvRunListener:
    """Appends every round's per-dojo TEST losses, and every checkpoint, to
    two CSV files that can be tailed or loaded with pandas during an
    overnight run.

    `rounds_path` gets one row per (round, dojo): phase, round_index, step,
    dojo, test_loss, status, quarantined. Long (one row per dojo) rather
    than wide (one column per dojo), since a phase's dojo set can differ
    from the next phase's. `checkpoints_path` (default: rounds_path with
    "_checkpoints" appended to its stem) gets one row per checkpoint:
    phase, round_index, step, path, encoder_path. Both files start with a
    header row on first write and are opened in append mode thereafter, so
    a run can resume writing into an existing log without truncating it.
    """

    def __init__(self, rounds_path: Path, checkpoints_path: Path | None = None) -> None:
        self._rounds_path = rounds_path
        self._checkpoints_path = checkpoints_path or rounds_path.with_name(
            rounds_path.stem + "_checkpoints" + rounds_path.suffix
        )
        self._rounds_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoints_path.parent.mkdir(parents=True, exist_ok=True)

    def on_round_end(self, report: RoundReport) -> None:
        """Append one CSV row per dojo scored this round.

        Input: RoundReport. Output: None. Side effects: appends to
        rounds_path, creating it (with a header) on first call.
        Exceptions: OSError on write failure (the Trainer logs and skips
        it, same as any other listener call)."""
        is_new = not self._rounds_path.exists()
        with self._rounds_path.open("a", newline="") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(
                    [
                        "phase",
                        "round_index",
                        "step",
                        "dojo",
                        "test_loss",
                        "status",
                        "quarantined",
                    ]
                )
            for name, loss in report.per_dojo_test_loss.items():
                status = report.statuses.get(name)
                writer.writerow(
                    [
                        report.phase,
                        report.round_index,
                        report.step,
                        name,
                        loss,
                        status.value if status is not None else "",
                        name in report.quarantined,
                    ]
                )

    def on_checkpoint(self, record: CheckpointRecord) -> None:
        """Append one CSV row for this checkpoint.

        Input: CheckpointRecord. Output: None. Side effects: appends to
        checkpoints_path, creating it (with a header) on first call.
        Exceptions: OSError on write failure (same contract as
        on_round_end)."""
        is_new = not self._checkpoints_path.exists()
        with self._checkpoints_path.open("a", newline="") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(
                    ["phase", "round_index", "step", "path", "encoder_path"]
                )
            report = record.report
            writer.writerow(
                [
                    report.phase,
                    report.round_index,
                    report.step,
                    record.path,
                    record.encoder_path,
                ]
            )
