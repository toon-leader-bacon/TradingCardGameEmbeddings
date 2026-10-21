"""PrefetchingBatchStreams: build each round's scheduled TRAIN batches,
and reserve their cards in the feature cache, on a worker thread while
the GPU trains. The Trainer's BatchFeed when
HardwareLimits.prefetch_depth >= 1 (src/training/diet/batch_feed.py).

Decoupled access/execute: the trainer hands over each round's
RoundSchedule (begin_round), the worker builds the scheduled steps in
order, at most `depth` steps ahead of the trainer, and the trainer takes
them in order with next_batch(name). Each built batch is reserved
(BatchWarmer.reserve: its cards pinned in the cache's memory tiers, then
warmed), so nothing it needs is evicted before its step; next_batch
releases the previous step's reservation. The worker never builds past
the round's schedule, so it is idle during evaluation.

Only the worker advances each dojo's DojoBatchStream, and always in
order, so a dojo's batch sequence is the one it would yield without
prefetch, given that:
- its randomness is all its own seeded RNG, never the global `random`
  (no dojo draws from it at training time);
- its batches(TRAIN) reads no training-time state (weights, step counts):
  batches are built up to `depth` steps ahead;
- its compute_loss reads nothing batches(TRAIN) advances, since the two
  run on different threads at once. GenericDojo's and ContrastiveDojo's
  read only their loss object and head; a new Dojo must keep that.
The same batches need not give bit-identical losses: warming changes
which tier answers and the order rows are evicted in.

A reschedule (after a mid-round quarantine) keeps the built steps that
still match the new schedule's prefix; the rest move, released, to their
dojo's leftover pile, ordered by build sequence. When a dojo comes up
again the worker takes its oldest leftover instead of building, and
reserves it again, so no stream skips a batch. A quarantined dojo's
leftovers are dropped at close.

An error from a stream is queued in its step's place and re-raised to the
consumer in order (its own type, so the trainer's fault ledger classifies
it as before); the worker keeps going, and the stream's next call starts
a fresh pass, exactly as when DojoBatchStream is called directly. A
stopped or dead worker raises PrefetchStoppedError, and a call out of
schedule order ScheduleMismatchError, never a dojo's error.
"""

import bisect
import enum
import logging
import threading
from collections import deque
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Mapping, Protocol

import torch

from src.dojos.dojo import DojoBatch
from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.feature_cache.pooled_feature_cache import FeatureReservation
from src.encoder_model.precision import Precision, parameter_device
from src.schema.card import GenericCard
from src.schema.type_hints import iter_cards
from src.training.diet.batch_feed_errors import (
    PrefetchStoppedError,
    ScheduleMismatchError,
)
from src.training.diet.dojo_batch_stream import DojoBatchStream
from src.training.diet.round_schedule import RoundSchedule, require_next_step

_logger = logging.getLogger(__name__)

# Log the first warming failure, then every this-many-th
_WARM_FAILURE_LOG_EVERY = 100

# What a built batch holds when nothing is pinned for it
_NO_RESERVATION = FeatureReservation(())


class BatchWarmer(Protocol):
    """What the prefetch worker does with each built batch's cards."""

    def reserve(self, cards: list[GenericCard]) -> FeatureReservation:
        """Pin the cards in the feature cache's memory tiers, then warm
        them. Called on the worker thread only.
        Inputs: cards (one batch's; may be empty).
        Output: FeatureReservation, held until release.
        Side effects: see implementations.
        Exceptions: whatever warming raises (nothing stays pinned)."""
        ...

    def release(self, reservation: FeatureReservation) -> None:
        """Remove a reservation's pins. Any thread.
        Inputs: reservation (from reserve). Output: none.
        Side effects: its rows become evictable.
        Exceptions: ValueError if it was already released."""
        ...


class _Lifecycle(enum.Enum):
    """The worker's state. Transitions: NOT_STARTED -start-> RUNNING;
    any -close-> CLOSED; RUNNING -> FAILED when the worker loop itself
    dies. Rounds are separate: a RUNNING worker without a round is idle."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True)
class _BuiltBatch:
    """A built batch and the pins held for it (_NO_RESERVATION when none
    are: warming failed, or it waits in a leftover pile)."""

    batch: DojoBatch
    reservation: FeatureReservation


@dataclass(frozen=True)
class _BuildFailure:
    """What the stream raised while building a batch."""

    error: Exception


@dataclass(frozen=True)
class _BuiltStep:
    """One step's result, as queued for the trainer or kept as a leftover.

    name: its dojo.
    sequence: its place in the dojo's stream (0, 1, ...); leftovers are
        taken in this order, so a stream's batches are used in order.
    result: the batch with its pins, or the stream's error.
    """

    name: str
    sequence: int
    result: _BuiltBatch | _BuildFailure


@dataclass
class _InFlight:
    """The step the worker is building. wanted turns False when a
    reschedule or end_round leaves it off the schedule: it then lands in
    its dojo's leftover pile, released."""

    name: str
    wanted: bool = True


@dataclass(frozen=True)
class _Job:
    """What the worker was told to produce: name's next step, from its
    oldest leftover when it has one, else built fresh."""

    name: str
    leftover: _BuiltStep | None


class PrefetchingBatchStreams:
    """BatchFeed: each round's scheduled TRAIN batches, built and reserved
    ahead on one worker thread. A context manager: entering starts it,
    leaving closes it.

    begin_round, next_batch, reschedule, end_round and close are meant for
    one consumer thread (the Trainer's)."""

    def __init__(
        self,
        streams: Mapping[str, DojoBatchStream],
        warmer: BatchWarmer,
        depth: int,
    ) -> None:
        """Inputs: streams (dojo name -> its stream; only this object may
            call them from now on), warmer (reserves each built batch's
            cards on the worker thread), depth (>= 1, scheduled steps built
            ahead of the trainer).
        Output: none (constructor). Side effects: none; call start().
        Exceptions: ValueError if depth < 1 or streams is empty.

        Example:
            >>> prefetch = PrefetchingBatchStreams(streams, build_model_warmer(model, "fp16"), 2)
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if not streams:
            raise ValueError("PrefetchingBatchStreams needs at least one stream")
        self._streams = dict(streams)
        self._warmer = warmer
        self._depth = depth
        # One condition guards every field below. Rule: every state change
        # calls notify_all, and every wait sits in a predicate loop
        self._condition = threading.Condition()
        self._state = _Lifecycle.NOT_STARTED
        # The round's unconsumed steps, the trainer's next first; None
        # between rounds. _ready holds the built results of its first steps
        self._pending: list[str] | None = None
        self._ready: deque[_BuiltStep] = deque()
        self._in_flight: _InFlight | None = None
        # Per dojo: built steps taken off the schedule, oldest first, released
        self._leftovers: dict[str, list[_BuiltStep]] = {name: [] for name in streams}
        # Whatever killed the worker loop: set when it dies (the state becomes
        # FAILED unless already CLOSED) and kept through a later close
        self._fatal: BaseException | None = None
        self._warm_failures = 0
        self._worker: threading.Thread | None = None
        # Worker thread only: each dojo's next build sequence
        self._next_sequence: dict[str, int] = {name: 0 for name in streams}
        # Consumer thread only: the pins of the step last handed out
        self._taken = _NO_RESERVATION

    def __enter__(self) -> "PrefetchingBatchStreams":
        """Start the worker. Inputs: none. Output: self.
        Side effects: as start. Exceptions: as start."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close, however the block ends. Inputs: the exception, if any.
        Output: none (exceptions propagate). Side effects: as close.
        Exceptions: none."""
        self.close()

    @property
    def warm_failures(self) -> int:
        """Batches whose reservation raised (their cards were left to the
        training thread). Inputs: none. Output: int. Side effects: none.
        Exceptions: none."""
        with self._condition:
            return self._warm_failures

    def start(self) -> None:
        """NOT_STARTED -> RUNNING: start the daemon worker thread, idle
        until begin_round.

        Inputs: none. Output: none. Side effects: one thread starts.
        Exceptions: RuntimeError in any other state (started already, or
            closed).

        Example:
            >>> prefetch.start()
        """
        with self._condition:
            if self._state is not _Lifecycle.NOT_STARTED:
                raise RuntimeError(f"cannot start prefetching when {self._state.value}")
            self._state = _Lifecycle.RUNNING
            self._worker = threading.Thread(
                target=self._run_worker, name="batch-prefetch", daemon=True
            )
            self._worker.start()
            self._condition.notify_all()

    def begin_round(self, schedule: RoundSchedule) -> None:
        """See BatchFeed.begin_round: the worker starts on the schedule.

        Inputs: schedule (names from this feed's streams). Output: none.
        Side effects: sets the pending steps; wakes the worker.
        Exceptions: ScheduleMismatchError if a round is open;
            PrefetchStoppedError unless RUNNING (chained to the worker's
            error when FAILED).

        Example:
            >>> prefetch.begin_round(RoundSchedule(("a", "b", "a")))
        """
        with self._condition:
            if self._state is not _Lifecycle.RUNNING:
                raise self._stopped_error("cannot begin a round")
            if self._pending is not None:
                raise ScheduleMismatchError("begin_round before the last round ended")
            self._pending = list(schedule.dojo_names)
            self._condition.notify_all()

    def next_batch(self, name: str) -> DojoBatch:
        """See BatchFeed.next_batch: the next step's built batch, waiting
        for it while the worker is RUNNING.

        Inputs: name (str, the next scheduled dojo).
        Output: DojoBatch.
        Side effects: releases the previous step's reservation first;
            consumes the step, which frees room for the worker; holds the
            returned batch's reservation until the next call or end_round.
        Exceptions: ScheduleMismatchError (see require_next_step);
            PrefetchStoppedError if nothing is built and the worker is not
            RUNNING (chained to its fatal error when FAILED); whatever
            building that batch raised, re-raised here in order.

        Example:
            >>> batch = prefetch.next_batch("contrastive.gwent")
        """
        # The previous step has gathered its rows: its pins can go
        previous, self._taken = self._taken, _NO_RESERVATION
        self._warmer.release(previous)

        with self._condition:
            require_next_step(self._pending, name)
            # Wait for the worker only while it runs
            while not self._ready:
                if self._state is not _Lifecycle.RUNNING:
                    raise self._stopped_error(f"no batch built for {name!r}")
                self._condition.wait()
            step = self._ready.popleft()
            self._consume_pending_head()
            self._condition.notify_all()

        if isinstance(step.result, _BuildFailure):
            raise step.result.error
        self._taken = step.result.reservation
        return step.result.batch

    def reschedule(self, remaining: RoundSchedule) -> None:
        """See BatchFeed.reschedule: keep the built (and in-flight) steps
        matching remaining's prefix; the rest go, released, to leftovers.

        Inputs: remaining (the new rest of the round). Output: none.
        Side effects: replaces the pending steps; may mark the in-flight
            step unwanted; releases dropped steps' reservations.
        Exceptions: ScheduleMismatchError if no round is open.

        Example:
            >>> prefetch.reschedule(RoundSchedule(("b", "b")))
        """
        with self._condition:
            if self._pending is None:
                raise ScheduleMismatchError("reschedule with no round open")
            new_pending = list(remaining.dojo_names)
            # Keep the built (then in-flight) steps the new schedule starts with
            built_names = [step.name for step in self._ready]
            in_flight = self._in_flight
            if in_flight is not None and in_flight.wanted:
                built_names.append(in_flight.name)
            keep = _matching_prefix_length(built_names, new_pending)
            if in_flight is not None and keep < len(built_names):
                in_flight.wanted = False  # dropped too, or after a dropped step
            dropped = [self._ready.pop() for _ in range(len(self._ready) - keep)]
            to_release = self._move_to_leftovers(dropped)
            self._pending = new_pending
            self._condition.notify_all()
        self._release_each(to_release)

    def end_round(self) -> None:
        """See BatchFeed.end_round: stop the worker taking steps, wait for
        its in-flight step, move unused built steps to leftovers, and
        release every reservation. Does nothing between rounds.

        Inputs: none. Output: none.
        Side effects: blocks until the worker is idle (a hung dojo or warm
            therefore hangs it); releases reservations.
        Exceptions: none.

        Example:
            >>> prefetch.end_round()
        """
        previous, self._taken = self._taken, _NO_RESERVATION
        to_release = [previous]
        with self._condition:
            if self._pending is not None:
                # No step is wanted any more; wait for the one in flight
                self._pending = []
                if self._in_flight is not None:
                    self._in_flight.wanted = False
                while self._in_flight is not None:
                    self._condition.wait()
                dropped = list(self._ready)
                self._ready.clear()
                to_release += self._move_to_leftovers(dropped)
                self._pending = None
                self._condition.notify_all()
        self._release_each(to_release)

    def close(self) -> None:
        """Any state -> CLOSED; join the worker; release every reservation;
        drop built steps and leftovers. Idempotent. Call it before closing
        the model's feature cache: a worker still warming would reopen a
        closed SQLite tier.

        Inputs: none. Output: none.
        Side effects: blocks until the in-flight build finishes, then
            joins the thread; releases reservations.
        Exceptions: none.

        Example:
            >>> prefetch.close()
        """
        with self._condition:
            self._state = _Lifecycle.CLOSED
            self._condition.notify_all()
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join()

        # Nothing builds now: release what is held and forget the rest
        previous, self._taken = self._taken, _NO_RESERVATION
        to_release = [previous]
        with self._condition:
            to_release += self._move_to_leftovers(list(self._ready))
            self._ready.clear()
            self._pending = None
            for pile in self._leftovers.values():
                pile.clear()
            self._condition.notify_all()
        self._release_each(to_release)

    def _run_worker(self) -> None:
        """The worker loop: wait for the next scheduled step within depth
        (_wait_for_work), produce it outside the lock (_produce), land it
        (_land); until CLOSED. If the loop itself raises, the state becomes
        FAILED with the error kept, the in-flight mark is cleared and
        waiters are woken, so no consumer waits forever. Only a bug in the
        loop itself gets here (a step's own errors are kept as
        _BuildFailure), and a reservation it held then stays pinned; the
        run ends at the trainer's next feed call anyway.

        Private helper - single consumer is start (the thread's target).
        Inputs: none. Output: none.
        Side effects: advances streams, reserves through the warmer, fills
            _ready and the leftover piles.
        Exceptions: none escape the thread (kept in _fatal instead).
        """
        try:
            # Wait for a step, produce it outside the lock, land it
            while (job := self._wait_for_work()) is not None:
                step = self._produce(job)
                self._land(step)
        except BaseException as error:  # noqa: B036 - kept for next_batch, never lost
            with self._condition:
                self._fatal = error
                self._in_flight = None
                if self._state is not _Lifecycle.CLOSED:
                    self._state = _Lifecycle.FAILED
                self._condition.notify_all()

    def _wait_for_work(self) -> _Job | None:
        """Wait until RUNNING, nothing is in flight, and the next unbuilt
        step is within depth of the trainer and within the schedule; mark
        it in flight and return it, with its dojo's oldest leftover if any.
        None once CLOSED.

        Private helper - single consumer is _run_worker (takes the lock).
        Inputs: none. Output: _Job or None.
        Side effects: sets _in_flight; pops a leftover; waits.
        Exceptions: none.
        """
        with self._condition:
            while self._state is not _Lifecycle.CLOSED:
                name = self._next_unbuilt_step()
                if name is not None:
                    pile = self._leftovers[name]
                    leftover = pile.pop(0) if pile else None
                    self._in_flight = _InFlight(name)
                    self._condition.notify_all()
                    return _Job(name, leftover)
                self._condition.wait()
            return None

    def _next_unbuilt_step(self) -> str | None:
        """The dojo of the step to build now, if the worker may build: it
        is RUNNING, nothing is in flight, and the next unbuilt step exists
        and is within depth of the trainer.

        Private helper - single consumer is _wait_for_work (lock held).
        Inputs: none. Output: str or None. Side effects: none.
        Exceptions: none.
        """
        if self._state is not _Lifecycle.RUNNING or self._in_flight is not None:
            return None
        if self._pending is None:
            return None
        position = len(self._ready)
        if position >= self._depth or position >= len(self._pending):
            return None
        return self._pending[position]

    def _produce(self, job: _Job) -> _BuiltStep:
        """The job's step: a leftover failure as it is, a leftover batch
        reserved again, or a fresh batch built and reserved (with the
        dojo's next sequence).

        Private helper - single consumer is _run_worker (lock not held).
        Inputs: job. Output: _BuiltStep.
        Side effects: may advance the dojo's stream and its sequence;
            _reserve_cards.
        Exceptions: none (a stream's error becomes a _BuildFailure).
        """
        leftover = job.leftover
        # A leftover keeps its place in the stream; its batch is reserved again
        if leftover is not None:
            if isinstance(leftover.result, _BuildFailure):
                return leftover
            reservation = self._reserve_cards(job.name, leftover.result.batch)
            return replace(
                leftover, result=replace(leftover.result, reservation=reservation)
            )

        # Otherwise the stream's next batch, in its sequence
        sequence = self._next_sequence[job.name]
        self._next_sequence[job.name] = sequence + 1
        try:
            batch = self._streams[job.name].next_batch()
        except Exception as error:
            return _BuiltStep(job.name, sequence, _BuildFailure(error))
        reservation = self._reserve_cards(job.name, batch)
        return _BuiltStep(job.name, sequence, _BuiltBatch(batch, reservation))

    def _reserve_cards(self, name: str, batch: DojoBatch) -> FeatureReservation:
        """Reserve the batch's cards (flattened with type_hints.iter_cards).
        A failure is counted (warm_failures), logged on the first
        occurrence and every 100th after, and gives _NO_RESERVATION: the
        cards are then encoded on the training thread. (A CUDA error
        usually poisons the whole context, so the training thread fails
        too; that surfaces there.)

        Private helper - single consumer is _produce (lock not held).
        Inputs: name (for the log), batch. Output: FeatureReservation.
        Side effects: warmer.reserve; counts and logs failures.
        Exceptions: none.
        """
        try:
            return self._warmer.reserve(list(iter_cards(batch.inputs)))
        except Exception as error:
            with self._condition:
                self._warm_failures += 1
                failures = self._warm_failures
            if failures == 1 or failures % _WARM_FAILURE_LOG_EVERY == 0:
                _logger.warning(
                    "prefetch could not warm a %s batch (%d so far): %r",
                    name,
                    failures,
                    error,
                )
            return _NO_RESERVATION

    def _land(self, step: _BuiltStep) -> None:
        """Queue the in-flight step for the trainer if it is still wanted;
        otherwise release it into its dojo's leftovers (dropped when
        CLOSED). Clears the in-flight mark.

        Private helper - single consumer is _run_worker (takes the lock).
        Inputs: step. Output: none.
        Side effects: appends to _ready or a leftover pile; releases an
            unwanted step's reservation (after the lock); wakes waiters.
        Exceptions: none.
        """
        to_release: list[FeatureReservation] = []
        with self._condition:
            wanted = self._in_flight is not None and self._in_flight.wanted
            if self._state is _Lifecycle.CLOSED:
                if isinstance(step.result, _BuiltBatch):
                    to_release = [step.result.reservation]
            elif wanted:
                self._ready.append(step)
            else:
                to_release = self._move_to_leftovers([step])
            self._in_flight = None
            self._condition.notify_all()
        self._release_each(to_release)

    def _move_to_leftovers(self, steps: list[_BuiltStep]) -> list[FeatureReservation]:
        """Put steps in their dojos' leftover piles, by sequence, with
        their reservations stripped; return those reservations for the
        caller to release once the lock is dropped.

        Private helper - consumers are reschedule, end_round, close and
        _land (lock held).
        Inputs: steps. Output: list[FeatureReservation].
        Side effects: inserts into leftover piles. Exceptions: none.
        """
        result: list[FeatureReservation] = []
        for step in steps:
            if isinstance(step.result, _BuiltBatch):
                result.append(step.result.reservation)
            bisect.insort(
                self._leftovers[step.name],
                _with_reservation_stripped(step),
                key=lambda kept: kept.sequence,
            )
        return result

    def _consume_pending_head(self) -> None:
        """Drop the step the trainer just took from the pending steps.

        Private helper - single consumer is next_batch (lock held; the
        schedule check passed). Inputs: none. Output: none.
        Side effects: as described. Exceptions: none.
        """
        assert self._pending  # require_next_step checked it
        self._pending.pop(0)

    def _release_each(self, reservations: list[FeatureReservation]) -> None:
        """warmer.release each reservation (empty ones are no-ops).

        Private helper - consumers are reschedule, end_round, close and
        _land (lock not held; next_batch releases its one directly).
        Inputs: reservations. Output: none.
        Side effects: unpins. Exceptions: ValueError from the warmer for a
            double release (a bug).
        """
        for reservation in reservations:
            self._warmer.release(reservation)

    def _stopped_error(self, what: str) -> PrefetchStoppedError:
        """A PrefetchStoppedError naming what failed and the worker's
        state, chained to the fatal error when there is one.

        Private helper - consumers are begin_round and next_batch (lock
        held). Inputs: what (str). Output: PrefetchStoppedError.
        Side effects: none. Exceptions: none.
        """
        result = PrefetchStoppedError(
            f"{what}: the prefetch worker is {self._state.value}"
        )
        # Chain only a real cause: setting __cause__ also hides any context
        if self._fatal is not None:
            result.__cause__ = self._fatal
        return result


def _matching_prefix_length(built: list[str], scheduled: list[str]) -> int:
    """How many leading names built and scheduled share.

    Private helper - single consumer is PrefetchingBatchStreams.reschedule.
    Inputs: built, scheduled. Output: int. Side effects: none.
    Exceptions: none.
    """
    result = 0
    for built_name, scheduled_name in zip(built, scheduled):
        if built_name != scheduled_name:
            break
        result += 1
    return result


def _with_reservation_stripped(step: _BuiltStep) -> _BuiltStep:
    """step with _NO_RESERVATION in place of its batch's reservation (a
    failure is returned unchanged).

    Private helper - single consumer is _move_to_leftovers.
    Inputs: step. Output: _BuiltStep. Side effects: none. Exceptions: none.
    """
    if isinstance(step.result, _BuildFailure):
        return step
    return replace(step, result=replace(step.result, reservation=_NO_RESERVATION))


class _NoBatchWarming:
    """BatchWarmer that pins and warms nothing (Null Object): prefetch for
    a model without a feature cache still builds batches ahead."""

    def reserve(self, cards: list[GenericCard]) -> FeatureReservation:
        """An empty reservation. Inputs: cards (ignored).
        Output: FeatureReservation. Side effects: none. Exceptions: none."""
        return FeatureReservation(())

    def release(self, reservation: FeatureReservation) -> None:
        """Nothing to release. Inputs: reservation (ignored). Output: none.
        Side effects: none. Exceptions: none."""


NO_BATCH_WARMING: BatchWarmer = _NoBatchWarming()


def build_model_warmer(model: CardEncoderModel, precision: Precision) -> BatchWarmer:
    """A BatchWarmer that reserves a batch's cards in model's feature
    cache on the worker's own CUDA stream (when on CUDA), with the cache's
    training-miss durability (so random-mod texts stay off the disk under
    `write: warm_only`).

    Inputs: model (built with a feature cache), precision (the run's).
    Output: BatchWarmer (reserve only from one worker thread).
    Side effects: none until called; then pins and encodes the cards the
        cache lacks, on a stream it creates on first use.
    Exceptions: ValueError now, not per batch, if model has no feature
        cache or precision is not its compute precision (a configuration
        error no batch can recover from).

    Example:
        >>> warmer = build_model_warmer(model, "fp16")
    """
    cache = model.card_pipeline.feature_cache
    if cache is None:
        raise ValueError("prefetch warming needs a model built with a feature cache")
    if cache.fingerprint.compute_precision != precision:
        raise ValueError(
            f"the feature cache was built for {cache.fingerprint.compute_precision}, "
            f"not {precision}"
        )
    return _StreamModelWarmer(model, precision)


class _StreamModelWarmer:
    """BatchWarmer: model.reserve on the calling worker's own CUDA stream
    (made on the first CUDA call), with the cache's training-miss
    durability. Warns once when a memory tier first refuses rows because
    pinned rows fill it (lower prefetch_depth or grow the tier)."""

    def __init__(self, model: CardEncoderModel, precision: Precision) -> None:
        """Inputs: model (with a feature cache for precision), precision.
        Output: none (constructor). Side effects: none. Exceptions: none."""
        self._model = model
        self._precision = precision
        self._stream: torch.cuda.Stream | None = None
        self._warned_refused = False

    def reserve(self, cards: list[GenericCard]) -> FeatureReservation:
        """Pin and warm the cards.

        Inputs: cards (one batch's; may be empty).
        Output: FeatureReservation.
        Side effects: model.reserve (durable=None) on this warmer's stream
            when the model is on CUDA, else on the CPU; may log a warning.
        Exceptions: whatever model.reserve raises.

        Example:
            >>> reservation = warmer.reserve(cards)
        """
        if not cards:
            return _NO_RESERVATION
        device = parameter_device(self._model)
        if device.type != "cuda":
            result = self._model.reserve(cards, self._precision, durable=None)
        else:
            if self._stream is None:
                self._stream = torch.cuda.Stream(device)
            with torch.cuda.stream(self._stream):
                result = self._model.reserve(cards, self._precision, durable=None)
        self._warn_once_if_refused()
        return result

    def release(self, reservation: FeatureReservation) -> None:
        """model.release. Inputs: reservation. Output: none.
        Side effects: unpins. Exceptions: as model.release.

        Example:
            >>> warmer.release(reservation)
        """
        self._model.release(reservation)

    def _warn_once_if_refused(self) -> None:
        """Log a warning the first time any tier's refused_for_pins is
        above zero.

        Private helper - single consumer is reserve.
        Inputs: none. Output: none. Side effects: may log once.
        Exceptions: none.
        """
        if self._warned_refused:
            return
        stats = self._model.feature_cache_stats()
        if stats is None:
            return
        refused = [tier for tier in stats.tiers if tier.refused_for_pins]
        for tier in refused:
            _logger.warning(
                "the %s feature tier is full of pinned rows and refused %d rows; "
                "lower hardware.prefetch_depth or grow the tier",
                tier.name,
                tier.refused_for_pins,
            )
        self._warned_refused = bool(refused)
