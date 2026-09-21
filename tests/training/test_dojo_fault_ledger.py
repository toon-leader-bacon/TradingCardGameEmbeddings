import logging

import pytest

from src.training.diet.dojo_fault_ledger import DojoFaultLedger
from src.training.plan import FaultPolicy


def _ledger(dojo_limit: int = 2, overall_limit: int = 3) -> DojoFaultLedger:
    policy = FaultPolicy(dojo_limit, overall_limit)
    return DojoFaultLedger(policy, ("a", "b"))


def test_quarantines_a_dojo_after_consecutive_failures() -> None:
    ledger = _ledger()
    ledger.record_failure("a", RuntimeError("x"))
    assert not ledger.is_quarantined("a")
    ledger.record_failure("a", RuntimeError("x"))
    assert ledger.is_quarantined("a")
    assert ledger.quarantined_names() == frozenset({"a"})


def test_a_success_resets_the_consecutive_counts() -> None:
    ledger = _ledger()
    ledger.record_failure("a", RuntimeError("x"))
    ledger.record_success("a")
    ledger.record_failure("a", RuntimeError("x"))
    assert not ledger.is_quarantined("a")


def test_gives_up_after_consecutive_failures_across_dojos() -> None:
    ledger = _ledger(dojo_limit=10, overall_limit=3)
    for name in ("a", "b", "a"):
        assert not ledger.gave_up()
        ledger.record_failure(name, RuntimeError("x"))
    assert ledger.gave_up()


def test_logs_the_failure(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _ledger().record_failure("a", RuntimeError("boom"))
    assert "'a'" in caplog.text


def test_a_failure_with_no_dojo_counts_only_toward_giving_up() -> None:
    ledger = _ledger(overall_limit=2)
    ledger.record_failure(None, RuntimeError("sampler"))
    ledger.record_failure(None, RuntimeError("sampler"))
    assert ledger.gave_up() and ledger.quarantined_names() == frozenset()


def test_total_failures_quarantine_even_with_successes_between() -> None:
    ledger = DojoFaultLedger(FaultPolicy(100, 100, max_total_dojo_failures=2), ("a",))
    ledger.record_failure("a", RuntimeError("x"))
    ledger.record_success("a")
    ledger.record_failure("a", RuntimeError("x"))
    assert ledger.is_quarantined("a")
