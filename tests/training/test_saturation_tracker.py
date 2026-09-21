from src.training.diet.saturation_tracker import SaturationTracker
from src.training.recording.reports import DojoStatus
from tests.training.fakes import saturation_spec


def _tracker(**overrides: float) -> SaturationTracker:
    return SaturationTracker(saturation_spec(**overrides), ("a", "b"))


def test_all_dojos_start_active_and_the_phase_is_not_done() -> None:
    tracker = _tracker()
    assert tracker.active_dojo_names() == ["a", "b"]
    assert not tracker.phase_done()


def test_a_dojo_saturates_after_patience_rounds_without_improvement() -> None:
    tracker = _tracker(patience_rounds=2)
    tracker.record_round({"a": 1.0, "b": 1.0})  # first loss sets the best
    tracker.record_round({"a": 0.5, "b": 1.0})  # a improves, b stalls (1)
    statuses = tracker.record_round({"a": 0.4, "b": 1.0})  # b stalls (2)
    assert statuses == {"a": DojoStatus.ACTIVE, "b": DojoStatus.SATURATED}
    assert tracker.active_dojo_names() == ["a"]


def test_gains_smaller_than_epsilon_count_as_stagnation() -> None:
    tracker = SaturationTracker(saturation_spec(epsilon=0.1, patience_rounds=2), ("a",))
    tracker.record_round({"a": 1.0})
    tracker.record_round({"a": 0.95})
    statuses = tracker.record_round({"a": 0.9})  # 0.1 below 1.0 is not > epsilon
    assert statuses["a"] is DojoStatus.SATURATED


def test_a_saturated_dojo_reactivates_when_its_loss_regresses() -> None:
    tracker = SaturationTracker(
        saturation_spec(patience_rounds=1, reactivation_delta=0.5), ("a",)
    )
    tracker.record_round({"a": 1.0})
    assert tracker.record_round({"a": 1.0})["a"] is DojoStatus.SATURATED
    assert tracker.record_round({"a": 1.2})["a"] is DojoStatus.SATURATED
    assert tracker.record_round({"a": 1.6})["a"] is DojoStatus.ACTIVE


def test_phase_done_once_the_target_fraction_is_saturated() -> None:
    tracker = _tracker(patience_rounds=1, target_saturated_fraction=0.5)
    tracker.record_round({"a": 1.0, "b": 1.0})
    tracker.record_round({"a": 0.5, "b": 1.0})
    assert tracker.phase_done()


def test_a_dojo_missing_from_a_round_keeps_its_state_and_extras_are_ignored() -> None:
    tracker = _tracker(patience_rounds=1)
    tracker.record_round({"a": 1.0, "b": 1.0})
    statuses = tracker.record_round({"a": 0.5, "held_out": 3.0})
    assert statuses == {"a": DojoStatus.ACTIVE, "b": DojoStatus.ACTIVE}


def test_quarantined_dojos_count_as_saturated_for_phase_done() -> None:
    tracker = _tracker(target_saturated_fraction=1.0)
    assert not tracker.phase_done()
    assert tracker.phase_done(frozenset({"a", "b"}))
