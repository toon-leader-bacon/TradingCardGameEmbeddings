"""RunProfile: where a training run reports how long its work took.

The Trainer and the objects it builds (each dojo's batch stream, the
prefetch feed, the TEST pass) report one record per unit of work. The
default is NO_RUN_PROFILE (Null Object), which drops everything, so an
unprofiled run pays nothing but the call. CsvRunProfile writes one CSV
per record type, flushed every row, so a crashed run keeps its rows:

    <directory>/
        steps.csv        one row per training step (StepTiming)
        builds.csv       one row per TRAIN batch built (BuildTiming)
        reserves.csv     one row per prefetch reservation (ReserveTiming)
        evaluation.csv   one row per (round, dojo) TEST pass (EvaluationTiming)
        marks.csv        startup stages and round boundaries (mark)

Every record carries the StageClock deltas of the thread that did the
work (src/utils/stage_clock.py), so a step's mods, serialization, cache
lookup and live encoding time are in its row: one "<stage>_s" column
per ProfileStage, then live_texts. Times are seconds; `t` is seconds
since the profile was opened (the run sampler's samples.csv uses the
same clock, see seconds_since_open), and marks.csv also has wall time.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, cast

from src.schema.type_hints import BatchedTrainingInput, iter_cards
from src.training.recording.csv_table import CsvTable
from src.utils.stage_clock import ProfileStage, StageTotals

_logger = logging.getLogger(__name__)

_STEPS_CSV = "steps.csv"
_BUILDS_CSV = "builds.csv"
_RESERVES_CSV = "reserves.csv"
_EVALUATION_CSV = "evaluation.csv"
_MARKS_CSV = "marks.csv"


@dataclass(frozen=True)
class StepTiming:
    """One training step attempt, on the training thread.

    round_index, step_index: where it fell (step_index counts every
        attempt since the run started). dojo: its dojo. cards: cards in
        its batch (0 if it failed before the batch arrived). completed:
        whether it ran to the end (an fp16 overflow skip counts, as in
        Trainer._take_step); False for a skipped or failed step.
    wait_seconds: waiting for the batch (with prefetch: for the worker;
        without: building it). forward_seconds: the model and the loss.
    total_seconds: the whole step; the remainder (backward, clipping, the
        optimizer and bookkeeping) is total - wait - forward.
    stages: the training thread's StageClock deltas over the step.
    """

    round_index: int
    step_index: int
    dojo: str
    cards: int
    completed: bool
    wait_seconds: float
    forward_seconds: float
    total_seconds: float
    stages: StageTotals


@dataclass(frozen=True)
class BuildTiming:
    """One TRAIN batch built by a dojo's stream, on whichever thread built it.

    dojo, examples (len(batch)), cards, pass_index (how many passes over
    the dojo's TRAIN split the stream has started, 1 for the first),
    seconds (the whole build), thread (its name), stages (that thread's
    deltas over the build).
    """

    dojo: str
    examples: int
    cards: int
    pass_index: int
    seconds: float
    thread: str
    stages: StageTotals


@dataclass(frozen=True)
class ReserveTiming:
    """One built batch's cards reserved in the feature cache (prefetch
    worker): dojo, cards, seconds, stages (the worker's deltas: the live
    encoding of cacheable cold misses shows here)."""

    dojo: str
    cards: int
    seconds: float
    stages: StageTotals


@dataclass(frozen=True)
class EvaluationTiming:
    """One dojo's TEST pass in one round: round_index, dojo, examples,
    cards, seconds, completed (False: the pass failed and the dojo was
    omitted), stages (the evaluating thread's deltas)."""

    round_index: int
    dojo: str
    examples: int
    cards: int
    seconds: float
    completed: bool
    stages: StageTotals


class RunProfile(Protocol):
    """Sink for a run's timing records. Implementations must be
    thread-safe (steps arrive from the training thread, builds and
    reservations from the prefetch worker) and must never raise: a
    failing write must not stop training."""

    def record_step(self, timing: StepTiming) -> None:
        """Inputs: timing. Output: none. Side effects: see implementations.
        Exceptions: none."""
        ...

    def record_build(self, timing: BuildTiming) -> None:
        """As record_step, for a built batch."""
        ...

    def record_reserve(self, timing: ReserveTiming) -> None:
        """As record_step, for a reservation."""
        ...

    def record_evaluation(self, timing: EvaluationTiming) -> None:
        """As record_step, for one dojo's TEST pass."""
        ...

    def mark(self, text: str) -> None:
        """Note a moment (a startup stage starting or ending, a round
        boundary). Inputs: text. Output: none.
        Side effects: see implementations. Exceptions: none."""
        ...

    def close(self) -> None:
        """Flush and stop accepting records. Idempotent. Exceptions: none."""
        ...


class _NoRunProfile:
    """RunProfile that drops every record (Null Object)."""

    def record_step(self, timing: StepTiming) -> None:
        """Drops it."""

    def record_build(self, timing: BuildTiming) -> None:
        """Drops it."""

    def record_reserve(self, timing: ReserveTiming) -> None:
        """Drops it."""

    def record_evaluation(self, timing: EvaluationTiming) -> None:
        """Drops it."""

    def mark(self, text: str) -> None:
        """Drops it."""

    def close(self) -> None:
        """Nothing to close."""


# The default: profiling off
NO_RUN_PROFILE: RunProfile = _NoRunProfile()


class CsvRunProfile:
    """RunProfile writing the CSVs in the module docstring. A failing write
    is logged once (later ones at debug) and dropped, never raised."""

    def __init__(self, directory: Path, opened_at: float | None = None) -> None:
        """Inputs: directory (created with parents; must not hold any of
            the five CSVs yet), opened_at (the time.monotonic() reading
            every row's t counts from; None: now).
        Output: none. Side effects: creates directory and the five CSVs
            with headers; marks "profile opened".
        Exceptions: FileExistsError (before creating anything) if a CSV
            already exists; OSError.

        Example:
            >>> profile = CsvRunProfile(run_directory / "profile")
        """
        # Validate inputs: refuse before creating any file
        names = (_STEPS_CSV, _BUILDS_CSV, _RESERVES_CSV, _EVALUATION_CSV, _MARKS_CSV)
        existing = [name for name in names if (directory / name).exists()]
        if existing:
            raise FileExistsError(f"{directory} already holds {existing}")
        directory.mkdir(parents=True, exist_ok=True)

        self._started = time.monotonic() if opened_at is None else opened_at
        self._failures_lock = threading.Lock()
        self._failures = 0
        stages = stage_field_names()
        self._steps = CsvTable(
            directory / _STEPS_CSV,
            ["t", "round", "step", "dojo", "cards", "completed", "wait_s",
             "forward_s", "remainder_s", "total_s", *stages],
        )  # fmt: skip
        self._builds = CsvTable(
            directory / _BUILDS_CSV,
            ["t", "dojo", "examples", "cards", "pass", "build_s", "thread", *stages],
        )
        self._reserves = CsvTable(
            directory / _RESERVES_CSV, ["t", "dojo", "cards", "reserve_s", *stages]
        )
        self._evaluation = CsvTable(
            directory / _EVALUATION_CSV,
            ["t", "round", "dojo", "examples", "cards", "total_s", "completed",
             *stages],
        )  # fmt: skip
        self._marks = CsvTable(directory / _MARKS_CSV, ["t", "wall", "mark"])
        self.mark("profile opened")

    def seconds_since_open(self) -> float:
        """Seconds since opened_at (the profile's opening by default): the
        `t` of every row.
        Inputs: none. Output: float. Side effects: none. Exceptions: none.

        Example:
            >>> profile.seconds_since_open()
            12.5
        """
        return time.monotonic() - self._started

    def record_step(self, timing: StepTiming) -> None:
        """See RunProfile.record_step: one steps.csv row.

        Example:
            >>> profile.record_step(timing)
        """
        remainder = timing.total_seconds - timing.wait_seconds - timing.forward_seconds
        self._write(
            self._steps,
            {
                "round": timing.round_index,
                "step": timing.step_index,
                "dojo": timing.dojo,
                "cards": timing.cards,
                "completed": int(timing.completed),
                "wait_s": timing.wait_seconds,
                "forward_s": timing.forward_seconds,
                "remainder_s": remainder,
                "total_s": timing.total_seconds,
                **stage_fields(timing.stages),
            },
        )

    def record_build(self, timing: BuildTiming) -> None:
        """See RunProfile.record_build: one builds.csv row.

        Example:
            >>> profile.record_build(timing)
        """
        self._write(
            self._builds,
            {
                "dojo": timing.dojo,
                "examples": timing.examples,
                "cards": timing.cards,
                "pass": timing.pass_index,
                "build_s": timing.seconds,
                "thread": timing.thread,
                **stage_fields(timing.stages),
            },
        )

    def record_reserve(self, timing: ReserveTiming) -> None:
        """See RunProfile.record_reserve: one reserves.csv row.

        Example:
            >>> profile.record_reserve(timing)
        """
        self._write(
            self._reserves,
            {
                "dojo": timing.dojo,
                "cards": timing.cards,
                "reserve_s": timing.seconds,
                **stage_fields(timing.stages),
            },
        )

    def record_evaluation(self, timing: EvaluationTiming) -> None:
        """See RunProfile.record_evaluation: one evaluation.csv row.

        Example:
            >>> profile.record_evaluation(timing)
        """
        self._write(
            self._evaluation,
            {
                "round": timing.round_index,
                "dojo": timing.dojo,
                "examples": timing.examples,
                "cards": timing.cards,
                "total_s": timing.seconds,
                "completed": int(timing.completed),
                **stage_fields(timing.stages),
            },
        )

    def mark(self, text: str) -> None:
        """See RunProfile.mark: one marks.csv row with the wall-clock time.

        Example:
            >>> profile.mark("build_dojos start")
        """
        self._write(
            self._marks, {"wall": time.strftime("%Y-%m-%dT%H:%M:%S"), "mark": text}
        )

    def close(self) -> None:
        """See RunProfile.close: marks "profile closed", closes every CSV.

        Example:
            >>> profile.close()
        """
        self.mark("profile closed")
        for table in (
            self._steps,
            self._builds,
            self._reserves,
            self._evaluation,
            self._marks,
        ):
            table.close()

    def _write(self, table: CsvTable, row: Mapping[str, object]) -> None:
        """Write row with its `t`; log and drop a failure (the first at
        warning, later ones at debug). Private helper - consumers are the
        record methods and mark. Inputs: table, row (without t).
        Output: none. Side effects: appends a row; may log.
        Exceptions: none."""
        try:
            table.write({"t": self.seconds_since_open(), **row})
        except Exception:
            with self._failures_lock:
                self._failures += 1
                first = self._failures == 1
            log = _logger.warning if first else _logger.debug
            log("run profile dropped a row", exc_info=True)


def card_count(inputs: object) -> int:
    """How many cards a batch's inputs hold (type_hints.iter_cards), for a
    timing record; 0 for inputs that are not cards, so profiling never
    fails a step.

    Inputs: inputs (a DojoBatch's inputs). Output: int.
    Side effects: none. Exceptions: none.

    Example:
        >>> card_count(batch.inputs)
        512
    """
    try:
        return sum(1 for _ in iter_cards(cast(BatchedTrainingInput, inputs)))
    except TypeError:
        return 0


def stage_field_names() -> list[str]:
    """The stage columns a record's row ends with: "<stage>_s" per
    ProfileStage, then "live_texts".

    Inputs: none. Output: list[str]. Side effects: none. Exceptions: none.

    Example:
        >>> stage_field_names()[-1]
        'live_texts'
    """
    return [f"{stage.value}_s" for stage in ProfileStage] + ["live_texts"]


def stage_fields(stages: StageTotals) -> dict[str, float | int]:
    """stages as the columns stage_field_names names.

    Inputs: stages. Output: dict. Side effects: none. Exceptions: none.

    Example:
        >>> stage_fields(StageTotals.zero())["live_texts"]
        0
    """
    result: dict[str, float | int] = {
        f"{stage.value}_s": stages.seconds_in(stage) for stage in ProfileStage
    }
    result["live_texts"] = stages.live_texts
    return result
