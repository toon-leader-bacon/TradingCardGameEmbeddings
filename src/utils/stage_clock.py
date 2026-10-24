"""StageClock: per-thread running seconds spent in named hot-path stages.

Process-wide, like src/run_logging.py: the stages it times sit deep in
dojos/ and encoder_model/ (mods, serialization, cache lookups, live
encoding), where passing a profiler down every call would touch dozens
of signatures for a diagnostic. Off by default, so a stage costs a
context manager and one flag check; scripts/run_training.py --profile
turns it on.

A reader takes totals() before and after some work on the same thread
and subtracts (StageTotals.minus): the difference is what that thread
spent in each stage meanwhile. A stage nested inside itself (a mod
pipeline calling another) is counted once, at its outermost call.

GPU time: a stage that queues GPU work (live encoding) synchronizes the
calling thread's current CUDA stream on entry and on exit, when enabled
with synchronize_cuda. The entry sync charges earlier queued work to
whoever ran before, not to the stage. Both threads of a run (trainer and
prefetch worker) use the device's default stream, so a sync also waits
for the other thread's queued kernels: read GPU stage times as upper
bounds, and expect a profiled run to stall slightly more than an
unprofiled one.

Example:
    >>> STAGE_CLOCK.enable(synchronize_cuda=True)
    >>> before = STAGE_CLOCK.totals()
    >>> with STAGE_CLOCK.timed(ProfileStage.MODS):
    ...     batch = pipeline.apply(batch)
    >>> STAGE_CLOCK.totals().minus(before).seconds_in(ProfileStage.MODS)
    0.0012
"""

import contextlib
import enum
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping

import torch


class ProfileStage(enum.Enum):
    """A hot-path stage the clock times.

    MODS: a ModPipeline's apply or apply_single (augmentations and task
        mods). SERIALIZATION: cards to JSON texts before encoding.
    CACHE_LOOKUP: a feature cache call (features or reserve), live
        encoding of its misses included. LIVE_ENCODING: the frozen stages
        (ModernBERT, enrichment, pooling) on texts: cache misses, or every
        text without a cache.
    """

    MODS = "mods"
    SERIALIZATION = "serialization"
    CACHE_LOOKUP = "cache_lookup"
    LIVE_ENCODING = "live_encoding"


@dataclass(frozen=True)
class StageTotals:
    """One thread's running totals at one moment; every ProfileStage is
    present in seconds (build with of or zero).

    seconds: per stage, wall-clock seconds inside it (outermost calls).
    live_texts: texts the frozen stages encoded on this thread.
    """

    seconds: Mapping[ProfileStage, float]
    live_texts: int

    def __post_init__(self) -> None:
        # Validate inputs: every stage present, so readers never guess
        missing = [stage for stage in ProfileStage if stage not in self.seconds]
        if missing:
            raise ValueError(f"StageTotals lacks stages {missing}")

    @classmethod
    def of(
        cls, seconds: Mapping[ProfileStage, float], live_texts: int
    ) -> "StageTotals":
        """Totals from a snapshot of seconds (copied, read-only).

        Inputs: seconds (every stage), live_texts. Output: StageTotals.
        Side effects: none. Exceptions: ValueError if a stage is missing.

        Example:
            >>> StageTotals.of({stage: 0.0 for stage in ProfileStage}, 0)
        """
        return cls(MappingProxyType(dict(seconds)), live_texts)

    @classmethod
    def zero(cls) -> "StageTotals":
        """Totals with nothing timed. Inputs: none. Output: StageTotals.
        Side effects: none. Exceptions: none.

        Example:
            >>> StageTotals.zero().live_texts
            0
        """
        return cls.of({stage: 0.0 for stage in ProfileStage}, 0)

    def seconds_in(self, stage: ProfileStage) -> float:
        """Seconds spent in stage. Inputs: stage. Output: float.
        Side effects: none. Exceptions: none.

        Example:
            >>> timing.stages.seconds_in(ProfileStage.MODS)
            0.004
        """
        return self.seconds[stage]

    def minus(self, earlier: "StageTotals") -> "StageTotals":
        """What was added since earlier (same thread).

        Inputs: earlier (an older totals() of the same thread).
        Output: StageTotals. Side effects: none. Exceptions: none.

        Example:
            >>> STAGE_CLOCK.totals().minus(before).live_texts
            12
        """
        return StageTotals.of(
            {
                stage: self.seconds[stage] - earlier.seconds[stage]
                for stage in ProfileStage
            },
            self.live_texts - earlier.live_texts,
        )


class _ThreadTotals(threading.local):
    """The calling thread's running sums and nesting depths."""

    def __init__(self) -> None:
        self.seconds: dict[ProfileStage, float] = {stage: 0.0 for stage in ProfileStage}
        self.depth: dict[ProfileStage, int] = {stage: 0 for stage in ProfileStage}
        self.live_texts = 0


class StageClock:
    """Per-thread stage timer; see the module docstring."""

    def __init__(self) -> None:
        """Starts disabled. Inputs: none. Output: none. Side effects: none.
        Exceptions: none."""
        self._enabled = False
        self._synchronize_cuda = False
        self._threads = _ThreadTotals()

    @property
    def enabled(self) -> bool:
        """Whether stages are being timed. Inputs: none. Output: bool.
        Side effects: none. Exceptions: none."""
        return self._enabled

    def enable(self, synchronize_cuda: bool) -> None:
        """Start timing, from every thread.

        Inputs: synchronize_cuda (True: a stage that queues GPU work
            synchronizes the calling thread's current CUDA stream on entry
            and exit; see the module docstring for what that counts).
        Output: none. Side effects: sets the process-wide flags.
        Exceptions: none.

        Example:
            >>> STAGE_CLOCK.enable(synchronize_cuda=torch.cuda.is_available())
        """
        self._synchronize_cuda = synchronize_cuda and torch.cuda.is_available()
        self._enabled = True

    def disable(self) -> None:
        """Stop timing (totals are kept). Inputs: none. Output: none.
        Side effects: clears the flag. Exceptions: none.

        Example:
            >>> STAGE_CLOCK.disable()
        """
        self._enabled = False

    @contextlib.contextmanager
    def timed(
        self, stage: ProfileStage, queues_gpu_work: bool = False
    ) -> Iterator[None]:
        """Add the body's wall-clock time to this thread's stage total,
        unless the stage is already open on this thread (nested call).

        Inputs: stage, queues_gpu_work (True: synchronize on entry and exit
            of the outermost call, when enabled with synchronize_cuda).
        Output: a context manager.
        Side effects: when enabled, grows this thread's total for stage;
            may synchronize the CUDA stream. Nothing when disabled.
        Exceptions: whatever the body raises (the time is still added).

        Example:
            >>> with STAGE_CLOCK.timed(ProfileStage.SERIALIZATION):
            ...     texts = [serialize_card_to_json(card) for card in cards]
        """
        # Disabled: no clock reads at all
        if not self._enabled:
            yield
            return
        # Time only the outermost call of this stage on this thread
        threads = self._threads
        outermost = threads.depth[stage] == 0
        threads.depth[stage] += 1
        synchronize = outermost and queues_gpu_work and self._synchronize_cuda
        if synchronize:
            torch.cuda.current_stream().synchronize()
        start = time.perf_counter()
        completed = False
        try:
            yield
            completed = True
        finally:
            threads.depth[stage] -= 1
            if outermost:
                # After a failed body (an OOM, a CUDA error) a sync could
                # raise and hide the original error
                if synchronize and completed:
                    torch.cuda.current_stream().synchronize()
                threads.seconds[stage] += time.perf_counter() - start

    def count_live_texts(self, count: int) -> None:
        """Add count to this thread's live-encoded texts (when enabled).
        Inputs: count (>= 0). Output: none. Side effects: grows the total.
        Exceptions: none.

        Example:
            >>> STAGE_CLOCK.count_live_texts(len(texts))
        """
        if self._enabled:
            self._threads.live_texts += count

    def totals(self) -> StageTotals:
        """A copy of this thread's running totals (zeros on a thread that
        timed nothing), safe to keep while the clock runs on.
        Inputs: none. Output: StageTotals. Side effects: none.
        Exceptions: none.

        Example:
            >>> STAGE_CLOCK.totals().seconds_in(ProfileStage.MODS)
            3.1
        """
        threads = self._threads
        return StageTotals.of(threads.seconds, threads.live_texts)


# The process-wide clock every hot path reports to
STAGE_CLOCK = StageClock()
