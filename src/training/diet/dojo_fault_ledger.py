"""Failure bookkeeping for an unattended run.

A step that fails (exception, out-of-memory, non-finite loss) is logged and
skipped rather than ending the run. The ledger counts failures so that one
bad dojo is quarantined (dropped from the diet for the rest of the phase)
and a run that is failing everywhere stops cleanly instead of spinning.
"""

import logging

from src.training.plan import FaultPolicy

logger = logging.getLogger(__name__)


class DojoFaultLedger:
    """Counts consecutive step failures per dojo and overall.

    Inputs (constructor): policy (FaultPolicy), dojo_names (the phase's dojos).
    """

    def __init__(self, policy: FaultPolicy, dojo_names: tuple[str, ...]) -> None:
        self._policy = policy
        self._consecutive_by_dojo = {name: 0 for name in dojo_names}
        self._total_by_dojo = {name: 0 for name in dojo_names}
        self._consecutive_overall = 0
        self._quarantined: set[str] = set()

    def record_failure(self, dojo_name: str | None, error: BaseException) -> None:
        """Log the error and count it; quarantine the dojo at the limit.

        Inputs: dojo_name (str, or None when no dojo was chosen yet, e.g.
            diet sampling failed: counts toward the overall limit only),
            error (the caught exception).
        Output: None.
        Side effects: logs a warning with the traceback; updates counts.
        Exceptions: none.

        Example:
            >>> ledger.record_failure("pick", ValueError("bad row"))
        """
        logger.warning("step failed for dojo %r", dojo_name, exc_info=error)
        self._consecutive_overall += 1
        if dojo_name is None:
            return
        self._consecutive_by_dojo[dojo_name] += 1
        self._total_by_dojo[dojo_name] += 1
        if dojo_name not in self._quarantined and (
            self._consecutive_by_dojo[dojo_name]
            >= self._policy.max_consecutive_dojo_failures
            or self._total_by_dojo[dojo_name] >= self._policy.max_total_dojo_failures
        ):
            self._quarantined.add(dojo_name)
            logger.error("quarantining dojo %r for the rest of the phase", dojo_name)

    def record_success(self, dojo_name: str) -> None:
        """Reset the dojo's and the overall consecutive-failure counts.

        Inputs: dojo_name (str). Output: None. Side effects: updates
        counts. Exceptions: none.
        """
        self._consecutive_by_dojo[dojo_name] = 0
        self._consecutive_overall = 0

    def is_quarantined(self, dojo_name: str) -> bool:
        """True if the dojo hit max_consecutive_dojo_failures this phase."""
        return dojo_name in self._quarantined

    def quarantined_names(self) -> frozenset[str]:
        """Every quarantined dojo (for the RoundReport)."""
        return frozenset(self._quarantined)

    def gave_up(self) -> bool:
        """True once overall consecutive failures hit
        max_consecutive_failures: the run is diverged or broken and should
        stop cleanly, keeping its best checkpoint."""
        return self._consecutive_overall >= self._policy.max_consecutive_failures
