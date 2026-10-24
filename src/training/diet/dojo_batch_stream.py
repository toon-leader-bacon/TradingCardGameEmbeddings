"""An endless TRAIN batch source for one dojo."""

import threading
import time
from typing import Iterator

from src.dojos.dojo import BatchBudget, Dojo, DojoBatch
from src.schema.splits import Split
from src.training.recording.run_profile import (
    NO_RUN_PROFILE,
    BuildTiming,
    RunProfile,
    card_count,
)
from src.utils.stage_clock import STAGE_CLOCK


class DojoBatchStream:
    """Hands out TRAIN batches one at a time, starting a fresh pass over
    the split whenever the previous one is exhausted.

    Inputs (constructor): dojo (Dojo), budget (BatchBudget), profile
        (where each built batch's BuildTiming goes; dropped by default).
    """

    def __init__(
        self, dojo: Dojo, budget: BatchBudget, profile: RunProfile = NO_RUN_PROFILE
    ) -> None:
        self._dojo = dojo
        self._budget = budget
        self._profile = profile
        self._batches: Iterator[DojoBatch] | None = None
        # Passes over the TRAIN split started so far
        self._passes = 0

    def next_batch(self) -> DojoBatch:
        """Return the next TRAIN batch, and report how long it took to the
        profile (only when the stage clock is on).

        Inputs: none.
        Output: DojoBatch.
        Side effects: advances (or restarts) the dojo's TRAIN iterator;
            records a BuildTiming when profiling.
        Exceptions: RuntimeError if the dojo yields no batches at all; any
            error the dojo raises mid-pass (the next call starts a fresh
            pass, so the fault ledger's total-failure limit is what stops a
            dojo that fails at the same row every pass).

        Example:
            >>> DojoBatchStream(dojo, budget).next_batch()
        """
        if not STAGE_CLOCK.enabled:
            return self._built_batch()
        # Time the build, with this thread's stage totals around it
        before = STAGE_CLOCK.totals()
        start = time.perf_counter()
        result = self._built_batch()
        self._profile.record_build(
            BuildTiming(
                dojo=self._dojo.name,
                examples=len(result),
                cards=card_count(result.inputs),
                pass_index=self._passes,
                seconds=time.perf_counter() - start,
                thread=threading.current_thread().name,
                stages=STAGE_CLOCK.totals().minus(before),
            )
        )
        return result

    def _built_batch(self) -> DojoBatch:
        """The next TRAIN batch, as next_batch describes, untimed.

        Private helper - single consumer is next_batch.
        Inputs: none. Output: DojoBatch.
        Side effects: advances (or restarts) the dojo's TRAIN iterator.
        Exceptions: as next_batch.
        """
        # Start a pass if none is open
        if self._batches is None:
            self._batches = self._fresh_pass()
        # Take the next batch; StopIteration means the pass ended, but any
        # other error means the dead generator must not be mistaken for that
        try:
            return next(self._batches)
        except StopIteration:
            pass
        except Exception:
            self._batches = None  # the next call starts a fresh pass
            raise
        self._batches = self._fresh_pass()
        batch = next(self._batches, None)
        if batch is None:
            raise RuntimeError(f"dojo {self._dojo.name!r} yielded no TRAIN batches")
        return batch

    def _fresh_pass(self) -> Iterator[DojoBatch]:
        """A new pass over the dojo's TRAIN split, counted.

        Private helper - single consumer is _built_batch.
        Inputs: none. Output: the dojo's batch iterator.
        Side effects: counts the pass. Exceptions: whatever batches raises.
        """
        self._passes += 1
        return self._dojo.batches(Split.TRAIN, self._budget)
