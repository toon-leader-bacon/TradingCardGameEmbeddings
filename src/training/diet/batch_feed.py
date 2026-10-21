"""BatchFeed: where the Trainer takes each phase's TRAIN batches from,
either built on the training thread when asked (DirectBatchFeed) or
built ahead on a worker thread (PrefetchingBatchStreams).

The Trainer owns one feed per phase. At the start of each round it
draws a RoundSchedule and hands it over with begin_round; it then asks
for each step's batch in schedule order. A quarantine mid-round redraws
the rest of the round (reschedule). end_round, before evaluation, leaves
every dojo to the training thread alone; close ends the phase.

Both feeds follow the schedule, so the schedule, not the feed variant
or its depth, defines what a run trains on: at the end of a round that
ran to the end, every dojo's stream has advanced by exactly the batches
the round used.
"""

from typing import Mapping, Protocol

from src.dojos.dojo import DojoBatch
from src.training.diet.batch_feed_errors import ScheduleMismatchError
from src.training.diet.dojo_batch_stream import DojoBatchStream
from src.training.diet.prefetching_batch_streams import (
    BatchWarmer,
    PrefetchingBatchStreams,
)
from src.training.diet.round_schedule import RoundSchedule, require_next_step


class BatchFeed(Protocol):
    """One phase's TRAIN batches, one round's schedule at a time.

    Implementations: DirectBatchFeed (depth 0) and PrefetchingBatchStreams.
    Meant for one consumer thread (the Trainer's).
    """

    def begin_round(self, schedule: RoundSchedule) -> None:
        """Start a round with these steps.
        Inputs: schedule (dojo names from the feed's streams). Output: none.
        Side effects: see implementations (the prefetching feed's worker
            starts building up to depth steps ahead).
        Exceptions: ScheduleMismatchError if the previous round was not
            ended; PrefetchStoppedError (prefetching only) if the worker
            died."""
        ...

    def next_batch(self, name: str) -> DojoBatch:
        """The next scheduled step's batch. First releases the previous
        step's reservation, on every path.
        Inputs: name (str, the next scheduled dojo). Output: DojoBatch.
        Side effects: consumes the step; advances that dojo's stream (or
            takes a built batch).
        Exceptions: ScheduleMismatchError if name is not the next scheduled
            dojo, the schedule is used up or no round is open; whatever
            building the batch raised, in order; PrefetchStoppedError
            (prefetching only) when no worker will build it."""
        ...

    def reschedule(self, remaining: RoundSchedule) -> None:
        """Replace every unconsumed step of the round.
        Inputs: remaining (the new rest of the round). Output: none.
        Side effects: see implementations (built steps that no longer
            match are kept for their dojo's next turn, released).
        Exceptions: ScheduleMismatchError if no round is open."""
        ...

    def end_round(self) -> None:
        """End the round, used up or not. Afterwards no thread but the
        caller touches a dojo, and every reservation is released. Does
        nothing when no round is open.
        Inputs: none. Output: none. Side effects: see implementations.
        Exceptions: none."""
        ...

    def close(self) -> None:
        """Stop for good and release the worker and any reservation.
        Idempotent. Inputs: none. Output: none.
        Side effects: see implementations. Exceptions: none."""
        ...


class DirectBatchFeed:
    """BatchFeed that builds each batch on the calling thread when asked,
    in schedule order. It warms and reserves nothing."""

    def __init__(self, streams: Mapping[str, DojoBatchStream]) -> None:
        """Inputs: streams (dojo name -> its stream).
        Output: none (constructor). Side effects: none. Exceptions: none.

        Example:
            >>> feed = DirectBatchFeed({"pick": stream})
        """
        self._streams = dict(streams)
        # The round's unconsumed steps, next first; None between rounds
        self._pending: list[str] | None = None

    def begin_round(self, schedule: RoundSchedule) -> None:
        """See BatchFeed.begin_round.

        Example:
            >>> feed.begin_round(RoundSchedule(("pick", "pick")))
        """
        if self._pending is not None:
            raise ScheduleMismatchError("begin_round before the last round ended")
        self._pending = list(schedule.dojo_names)

    def next_batch(self, name: str) -> DojoBatch:
        """See BatchFeed.next_batch: streams[name].next_batch(), after the
        schedule check.

        Example:
            >>> feed.next_batch("pick")
        """
        require_next_step(self._pending, name)
        assert self._pending is not None  # require_next_step checked it
        self._pending.pop(0)
        return self._streams[name].next_batch()

    def reschedule(self, remaining: RoundSchedule) -> None:
        """See BatchFeed.reschedule: nothing is built ahead, so only the
        pending steps change.

        Example:
            >>> feed.reschedule(RoundSchedule(("pick",)))
        """
        if self._pending is None:
            raise ScheduleMismatchError("reschedule with no round open")
        self._pending = list(remaining.dojo_names)

    def end_round(self) -> None:
        """See BatchFeed.end_round: forgets the pending steps.

        Example:
            >>> feed.end_round()
        """
        self._pending = None

    def close(self) -> None:
        """See BatchFeed.close: nothing to release."""


def open_batch_feed(
    streams: Mapping[str, DojoBatchStream], warmer: BatchWarmer, depth: int
) -> BatchFeed:
    """The feed for one phase: direct at depth 0, else a started
    PrefetchingBatchStreams building up to depth scheduled steps ahead
    (Factory Method over the two BatchFeed variants).

    Inputs: streams (dojo name -> its stream; the feed owns them from now
        on), warmer (reserves each built batch's cards on the worker;
        ignored at depth 0), depth (>= 0, HardwareLimits.prefetch_depth).
    Output: BatchFeed, already started; the caller must close it.
    Side effects: at depth >= 1, starts one daemon worker thread.
    Exceptions: ValueError if depth < 0 or streams is empty (from
        PrefetchingBatchStreams at depth >= 1).

    Example:
        >>> feed = open_batch_feed(streams, build_model_warmer(model, "fp16"), 4)
    """
    # Validate inputs
    if depth < 0:
        raise ValueError(f"prefetch depth must be >= 0, got {depth}")
    # Depth 0: build on the training thread
    if depth == 0:
        return DirectBatchFeed(streams)
    # Otherwise build ahead on one worker thread
    result = PrefetchingBatchStreams(streams, warmer, depth)
    result.start()
    return result
