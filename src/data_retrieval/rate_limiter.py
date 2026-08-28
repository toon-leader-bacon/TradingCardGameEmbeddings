"""A simple politeness pacer shared by data_retrieval sources.

Deliberately placed at this shared level (rather than private to
hearthstonejson/, its first consumer) — the class has zero HTTP or
source-specific coupling, and any future polite scraper (reddit/,
website/) is expected to need the same pacing behavior, so it's kept
here from the start rather than moved later. See
src/data_retrieval/README.md.
"""

import time


class RateLimiter:
    """Blocks callers so consecutive calls are spaced apart politely.

    Not tied to any HTTP request/response types — callers are expected
    to call wait() immediately before making a request, this class
    knows nothing about HTTP.
    """

    def __init__(self, requests_per_minute: float) -> None:
        """
        Inputs:
            requests_per_minute: maximum polite request rate. Must be
                greater than 0.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self.requests_per_minute = requests_per_minute
        self._min_interval_seconds = 60.0 / requests_per_minute
        self._last_call_monotonic: float | None = None

    def wait(self) -> None:
        """Block, if necessary, until it's been long enough since the
        last call to wait() to stay within requests_per_minute.

        The first call in a RateLimiter's lifetime never blocks.

        Inputs: none (uses internal last-call state).
        Output: none.
        Side effects: may sleep the calling thread; updates internal
            last-call timestamp.
        Exceptions: none.

        Example:
            >>> limiter = RateLimiter(requests_per_minute=12)
            >>> limiter.wait()  # returns immediately, first call
            >>> limiter.wait()  # blocks up to 5 seconds if called again immediately
        """
        now = time.monotonic()
        if self._last_call_monotonic is not None:
            elapsed = now - self._last_call_monotonic
            remaining = self._min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
                # Re-read the clock: the time spent sleeping must count
                # toward the next call's elapsed-time calculation, or
                # paced calls silently lose their spacing (the sleep
                # duration would otherwise vanish from the bookkeeping).
                now = time.monotonic()

        self._last_call_monotonic = now
