from src.training.recording.reports import (
    DojoStatus,
    RoundReport,
    is_better_round,
    mean_test_loss_over,
)


def _report(losses: dict[str, float]) -> RoundReport:
    return RoundReport("p", 0, 0, losses, {}, frozenset())


def test_mean_over_only_the_named_dojos() -> None:
    report = _report({"a": 1.0, "b": 3.0, "held_out": 100.0})
    assert mean_test_loss_over(report, ("a", "b")) == 2.0


def test_missing_dojos_are_ignored_and_none_present_is_infinite() -> None:
    assert mean_test_loss_over(_report({"a": 1.0}), ("a", "gone")) == 1.0
    assert mean_test_loss_over(_report({}), ("a",)) == float("inf")


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
