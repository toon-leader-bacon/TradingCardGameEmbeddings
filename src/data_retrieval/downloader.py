"""Shared base class for every src/data_retrieval source.

See plans/downloader_base_class.md for the design discussion this came
out of, and src/data_retrieval/README.md for this container's overall
scope (fetch and land raw files on disk, no parsing).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from src.data_retrieval.rate_limiter import RateLimiter


class Downloader(ABC):
    """Base class for every src/data_retrieval source.

    phase_1() and phase_2() are named for ORDER, not for what each
    does — the verbs available to describe either phase ("collect",
    "fetch", "download") apply equally to both, since phase_2 always
    consumes phase_1's own output as its input; no pair of verbs
    distinguishes them better than "which one runs first." Convention:

    - phase_1() does whatever a source's first fetch needs, with no
      dependency on any other call this class makes: a single static
      file, a fully self-contained paginated walk, or — for an
      id-list-then-detail source — just the id-list collection step.
    - phase_2() does further fetching that depends on phase_1()'s own
      output (typically: one detail request per id phase_1()
      collected). A source with nothing further to fetch does NOT
      override phase_2() at all — it inherits the default below.

    Every subclass accepts, at minimum, `rate_limiter` and
    `raw_data_dir` in its constructor — both optional, both defaulted
    here — and may declare further source-specific constructor
    arguments beyond these two (e.g. a source-specific URL override,
    or windowing/pagination parameters that must stay constant across
    a resumed walk — see SpireCodexRunDownloader).
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path]
    # Every concrete subclass must set this (expected to be a path
    # under data/raw, per src/README.md) — there is no usable default
    # at this base-class level, since it's source-specific by
    # definition. Not declared @abstractmethod (Python has no clean
    # "abstract ClassVar" enforcement) — a subclass that omits it fails
    # loudly the first time __init__ reads self.DEFAULT_RAW_DATA_DIR
    # (AttributeError), which is an acceptable substitute for a type
    # checker catching it at author time.

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: paces every outgoing request this downloader
                makes. Defaults to RateLimiter(requests_per_minute=60)
                when omitted — every external source in this container
                gets paced absent a source-specific reason to differ
                (this project's stated convention — see
                SpireCodexRunDownloader's module docstring).
            raw_data_dir: directory this downloader's output is written
                into. Defaults to self.DEFAULT_RAW_DATA_DIR when
                omitted.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/phase_2()
            are called.
        Exceptions: none.
        """
        self.rate_limiter = (
            rate_limiter
            if rate_limiter is not None
            else RateLimiter(requests_per_minute=60)
        )
        self.raw_data_dir = (
            raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR
        )
        self._phase_1_result: Path | None = None

    def phase_1(self) -> Path:
        """Run this downloader's first phase, caching its result for
        phase_2()'s default implementation.

        Concrete, not abstract — every subclass implements
        _run_phase_1() instead of overriding phase_1() directly, so
        the phase_2-cache bookkeeping happens by construction rather
        than relying on each subclass remembering to set it itself.

        Inputs: none (delegates to self._run_phase_1()).
        Output: _run_phase_1()'s result.
        Side effects: whatever _run_phase_1() does; stores its result
            on self.
        Exceptions: whatever _run_phase_1() raises.

        Example:
            >>> downloader.phase_1()
        """
        self._phase_1_result = self._run_phase_1()
        return self._phase_1_result

    @abstractmethod
    def _run_phase_1(self) -> Path:
        """Subclass hook for phase_1()'s actual work — see phase_1()
        and this class's docstring for what phase_1() is responsible
        for."""
        ...

    def phase_2(self) -> Path:
        """Default: no second phase. Returns phase_1()'s own cached
        result unchanged, doing no further work.

        A source whose second phase does real work overrides this
        method directly instead of relying on this default.

        Inputs: none (uses self._phase_1_result).
        Output: phase_1()'s cached result.
        Side effects: none.
        Exceptions: raises RuntimeError if phase_1() has not yet been
            called on this instance.

        Example:
            >>> downloader.phase_1()
            >>> downloader.phase_2()
        """
        if self._phase_1_result is None:
            raise RuntimeError(
                f"{type(self).__name__}.phase_2() called before phase_1()"
            )
        return self._phase_1_result
