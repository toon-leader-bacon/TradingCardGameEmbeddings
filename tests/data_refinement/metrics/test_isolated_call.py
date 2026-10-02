"""Tests for isolated_call.py's call_isolated."""

import logging

import pytest

from src.data_refinement.metrics.isolated_call import FAILURE_MARKER, call_isolated

_logger = logging.getLogger("test_isolated_call")


def test_a_successful_call_returns_true_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[int] = []

    with caplog.at_level(logging.ERROR):
        assert call_isolated(_logger, "metric", "accumulate()", lambda: calls.append(1))

    assert calls == [1]
    assert not caplog.records


def test_a_failing_call_returns_false_and_logs_one_loud_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode() -> None:
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR):
        assert not call_isolated(_logger, "MyMetric", "finalize() on A.csv", explode)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage().startswith(FAILURE_MARKER)
    assert "'MyMetric'" in record.getMessage()
    assert "finalize() on A.csv" in record.getMessage()
    assert record.exc_info is not None
