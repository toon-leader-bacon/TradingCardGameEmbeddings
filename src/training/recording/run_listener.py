"""Observer hooks on a training run, plus a reader for the rounds CSV that
CsvRunListener writes (the module that writes the format also reads it)."""

import csv
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from src.training.recording.reports import CheckpointRecord, DojoStatus, RoundReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundRow:
    """One row of a rounds CSV: one dojo's TEST loss in one round.

    The single definition of the rounds-CSV format: the header is the field
    names in declaration order, CsvRunListener writes RoundRows through
    _round_row_cells, and read_rounds_csv parses them back through
    _parse_round_row (its inverse). Adding a column means adding a field
    here and its parse in _parse_round_row (a missed parse fails loudly on
    the cell count).

    status: None where the CSV holds "" (a dojo scored but not in the
        phase's diet, e.g. a held-out dojo).
    """

    phase: str
    round_index: int
    step: int
    elapsed_seconds: float
    dojo: str
    test_loss: float
    status: DojoStatus | None
    quarantined: bool


# The rounds CSV header is derived from RoundRow; the reader accepts exactly it
_ROUNDS_HEADER = tuple(field.name for field in fields(RoundRow))
_CHECKPOINTS_HEADER = ("phase", "round_index", "step", "path", "encoder_path")


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

    `rounds_path` gets one row per (round, dojo): a RoundRow.
    Long (one row per dojo) rather than wide (one column per dojo), since a
    phase's dojo set can differ from the next phase's. `checkpoints_path`
    (default: rounds_path with "_checkpoints" appended to its stem) gets
    one row per checkpoint, columns _CHECKPOINTS_HEADER. Each file starts
    with a header row on first write and is appended to thereafter, so a
    run can resume writing into an existing log without truncating it.

    Constructing a listener has no side effect: parent directories are
    created on first write.
    """

    def __init__(self, rounds_path: Path, checkpoints_path: Path | None = None) -> None:
        self._rounds_path = rounds_path
        self._checkpoints_path = checkpoints_path or rounds_path.with_name(
            rounds_path.stem + "_checkpoints" + rounds_path.suffix
        )

    def on_round_end(self, report: RoundReport) -> None:
        """Append one CSV row per dojo scored this round.

        Input: RoundReport. Output: None.
        Side effects: appends to rounds_path, creating it (and its parent
            directories, with a header) on first call.
        Exceptions: OSError on write failure; ValueError if rounds_path
            exists with a different header (e.g. written before
            elapsed_seconds existed). The Trainer logs and skips both, same
            as any other listener call.
        """
        rows = [_round_row_cells(row) for row in _round_rows_of(report)]
        _append_rows(self._rounds_path, _ROUNDS_HEADER, rows)

    def on_checkpoint(self, record: CheckpointRecord) -> None:
        """Append one CSV row for this checkpoint.

        Input: CheckpointRecord. Output: None.
        Side effects: appends to checkpoints_path, creating it (and its
            parent directories, with a header) on first call.
        Exceptions: same as on_round_end.
        """
        report = record.report
        row = [
            report.phase,
            report.round_index,
            report.step,
            record.path,
            record.encoder_path,
        ]
        _append_rows(self._checkpoints_path, _CHECKPOINTS_HEADER, [row])


def read_rounds_csv(path: Path) -> list[RoundRow]:
    """Parse a rounds CSV written by CsvRunListener.

    Inputs: path (Path) to the rounds CSV.
    Output: list[RoundRow], in file order.
    Side effects: reads the file.
    Exceptions: FileNotFoundError if path is absent; ValueError if the
        header is not _ROUNDS_HEADER or a cell does not parse (the message
        names the line number).

    Example:
        >>> rows = read_rounds_csv(Path("runs/a/rounds.csv"))
        >>> rows[0].dojo, rows[0].test_loss
        ('pick', 1.37)
    """
    result: list[RoundRow] = []
    with path.open(newline="") as file:
        reader = csv.reader(file)
        # Validate the header before trusting any column position
        _require_header(path, next(reader, None), _ROUNDS_HEADER)
        # Parse each data row; line 1 is the header
        for line_number, cells in enumerate(reader, start=2):
            result.append(_parse_round_row(path, line_number, cells))
    return result


def _round_rows_of(report: RoundReport) -> list[RoundRow]:
    """One RoundRow per dojo scored in report, in per_dojo_test_loss order;
    status None for a dojo the report has no status for."""
    return [
        RoundRow(
            phase=report.phase,
            round_index=report.round_index,
            step=report.step,
            elapsed_seconds=report.elapsed_seconds,
            dojo=name,
            test_loss=loss,
            status=report.statuses.get(name),
            quarantined=name in report.quarantined,
        )
        for name, loss in report.per_dojo_test_loss.items()
    ]


def _round_row_cells(row: RoundRow) -> list[object]:
    """row's cells in field order: DojoStatus as its value, None status as
    "", everything else as is (csv.writer stringifies)."""
    cells: list[object] = []
    for field in fields(RoundRow):
        value = getattr(row, field.name)
        if isinstance(value, DojoStatus):
            value = value.value
        elif value is None:
            value = ""
        cells.append(value)
    return cells


def _append_rows(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]
) -> None:
    """Append rows to a CSV, first creating parent directories and writing
    `header` if the file is new; if it exists, its first row must equal
    `header`. Raises ValueError on a header mismatch (nothing appended),
    OSError on write failure."""
    header = tuple(header)
    if path.exists():
        with path.open(newline="") as file:
            _require_header(path, next(csv.reader(file), None), header)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(header)
        writer.writerows(rows)


def _require_header(
    path: Path, found: Sequence[str] | None, expected: Sequence[str]
) -> None:
    """Raise ValueError naming path unless found == expected (None = empty
    file)."""
    if found is None or tuple(found) != tuple(expected):
        raise ValueError(
            f"{path} has header {found}, expected {list(expected)}; "
            "write to a new file instead"
        )


def _parse_round_row(path: Path, line_number: int, cells: Sequence[str]) -> RoundRow:
    """One data row -> RoundRow; the inverse of _round_row_cells. ""
    status -> None; "True"/"False" -> quarantined. Raises ValueError
    naming path and line_number on a wrong cell count, an unparseable
    number, an unknown status, or a quarantined cell other than
    "True"/"False"."""
    if len(cells) != len(_ROUNDS_HEADER):
        raise ValueError(
            f"{path}:{line_number}: expected {len(_ROUNDS_HEADER)} cells, "
            f"got {len(cells)}"
        )
    phase, round_index, step, elapsed, dojo, loss, status, quarantined = cells
    try:
        return RoundRow(
            phase=phase,
            round_index=int(round_index),
            step=int(step),
            elapsed_seconds=float(elapsed),
            dojo=dojo,
            test_loss=float(loss),
            status=DojoStatus(status) if status else None,
            quarantined=_parse_bool_cell(quarantined),
        )
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error


def _parse_bool_cell(cell: str) -> bool:
    """ "True" -> True, "False" -> False (as csv.writer wrote them);
    ValueError otherwise."""
    if cell == "True":
        return True
    if cell == "False":
        return False
    raise ValueError(f"quarantined cell must be True or False, got {cell!r}")
