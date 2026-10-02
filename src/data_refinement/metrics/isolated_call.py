"""call_isolated - the one "run this metric step, log and swallow any
failure" helper, so one metric's bug never stops a scan for the others.

Shared by the 17lands chunk scanners and RowwiseMetric
(plans/seventeenlands_chunk_scan.md). The older per-source scanners
(sts_gg, play_gwent, isotropic, sts2_runs, and the draft and replay
row scanners) still carry their own copies of this try/log body; they
can move onto this helper as they are next touched.
"""

import logging
from typing import Callable

# The prefix of every isolated-failure log line: grep logs for it.
FAILURE_MARKER = "METRIC FAILURE"


def call_isolated(
    logger: logging.Logger, subject: object, step: str, call: Callable[[], object]
) -> bool:
    """Run call(), logging (not raising) any exception it raises.

    Inputs:
        logger: where a failure is logged.
        subject: what was being driven (a metric), named in the log line.
        step: which step failed, e.g. "accumulate()" or "finalize()".
        call: the zero-argument step to run.
    Output: True if call() returned, False if it raised.
    Side effects: call()'s; on failure, one ERROR log line starting
        with FAILURE_MARKER (greppable), naming subject and step, with
        the full traceback.
    Exceptions: none (every Exception is caught and logged).

    Example:
        >>> call_isolated(_logger, metric, "accumulate() on KTK.TradDraft.csv chunk 3",
        ...               partial(metric.accumulate, chunk))
        True
    """
    try:
        call()
    except Exception:
        logger.error(
            "%s: %r raised from %s; it is skipped for this step only, every "
            "other metric continues. Treat its output as untrusted.",
            FAILURE_MARKER,
            subject,
            step,
            exc_info=True,
        )
        return False
    return True
