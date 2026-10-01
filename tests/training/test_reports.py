import math

import pytest

from src.training.recording.reports import (
    DojoStatus,
    RoundReport,
    SplitLoss,
    is_better_round,
    mean_normalized_test_loss_over,
)


def _report(losses: dict[str, float], baseline: float = 1.0) -> RoundReport:
    """A report whose dojos all share one baseline (normalized = loss / it)."""
    split_losses = {name: SplitLoss(loss, baseline) for name, loss in losses.items()}
    return RoundReport("p", 0, 0, 0.0, split_losses, {}, frozenset())


def test_mean_over_only_the_named_dojos() -> None:
    report = _report({"a": 1.0, "b": 3.0, "held_out": 100.0})
    assert mean_normalized_test_loss_over(report, ("a", "b")) == 2.0


def test_missing_dojos_are_ignored_and_none_present_is_infinite() -> None:
    assert mean_normalized_test_loss_over(_report({"a": 1.0}), ("a", "gone")) == 1.0
    assert mean_normalized_test_loss_over(_report({}), ("a",)) == float("inf")


def test_the_mean_is_of_normalized_not_raw_losses() -> None:
    # A huge-scale dojo at half its baseline and a small one at its baseline
    report = RoundReport(
        "p",
        0,
        0,
        0.0,
        {"big": SplitLoss(500.0, 1000.0), "small": SplitLoss(2.0, 2.0)},
        {},
        frozenset(),
    )
    assert mean_normalized_test_loss_over(report, ("big", "small")) == 0.75


def test_status_values() -> None:
    assert DojoStatus.SATURATED.value == "saturated"


def test_a_round_is_compared_only_on_dojos_scored_in_both() -> None:
    incumbent = _report({"easy": 0.1, "hard": 5.0})
    # Omitting the hard dojo must not let a worse round win
    candidate = _report({"easy": 0.2})
    assert not is_better_round(candidate, incumbent, ("easy", "hard"))
    assert is_better_round(_report({"easy": 0.05}), incumbent, ("easy", "hard"))


def test_a_round_sharing_no_dojo_is_never_better() -> None:
    assert not is_better_round(_report({"a": 0.0}), _report({"b": 1.0}), ("a", "b"))


def test_a_large_scale_dojo_does_not_decide_the_better_round() -> None:
    # Raw sums would favor the incumbent (big drops 10 vs small's 0.9 rise);
    # normalized, small's relative gain outweighs big's
    incumbent = RoundReport(
        "p",
        0,
        0,
        0.0,
        {"big": SplitLoss(1000.0, 1000.0), "small": SplitLoss(1.0, 1.0)},
        {},
        frozenset(),
    )
    candidate = RoundReport(
        "p",
        1,
        0,
        0.0,
        {"big": SplitLoss(1010.0, 1000.0), "small": SplitLoss(0.1, 1.0)},
        {},
        frozenset(),
    )
    assert is_better_round(candidate, incumbent, ("big", "small"))


class TestSplitLoss:
    def test_normalized_is_loss_over_baseline(self) -> None:
        assert SplitLoss(loss=0.3, baseline_loss=0.6).normalized == 0.5

    def test_a_zero_loss_is_perfect(self) -> None:
        assert SplitLoss(loss=0.0, baseline_loss=2.0).normalized == 0.0

    @pytest.mark.parametrize("baseline", [0.0, -1.0, math.inf, math.nan])
    def test_rejects_an_unusable_baseline(self, baseline: float) -> None:
        with pytest.raises(ValueError, match="baseline_loss"):
            SplitLoss(loss=1.0, baseline_loss=baseline)
