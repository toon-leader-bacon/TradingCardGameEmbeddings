import pytest

from src.training.learning_rate_schedule import LearningRateSchedule


def test_the_default_is_a_constant_full_rate() -> None:
    schedule = LearningRateSchedule()
    assert [schedule.multiplier(step, 10) for step in (0, 5, 9, 50)] == [1.0] * 4


def test_the_warmup_rises_linearly_to_the_full_rate() -> None:
    schedule = LearningRateSchedule(warmup_steps=4)
    multipliers = [schedule.multiplier(step, 100) for step in range(6)]
    assert multipliers == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]


def test_the_cosine_falls_from_one_to_the_floor_at_the_last_step() -> None:
    schedule = LearningRateSchedule(decay="cosine", min_lr_fraction=0.1)
    assert schedule.multiplier(0, 101) == pytest.approx(1.0)
    assert schedule.multiplier(50, 101) == pytest.approx(0.55)
    assert schedule.multiplier(100, 101) == pytest.approx(0.1)
    # Past the budget it stays at the floor
    assert schedule.multiplier(500, 101) == pytest.approx(0.1)


def test_the_cosine_starts_after_the_warmup() -> None:
    schedule = LearningRateSchedule(warmup_steps=10, decay="cosine")
    assert schedule.multiplier(9, 111) == pytest.approx(1.0)
    assert schedule.multiplier(10, 111) == pytest.approx(1.0)
    assert schedule.multiplier(60, 111) == pytest.approx(0.5)
    assert schedule.multiplier(110, 111) == pytest.approx(0.0)


def test_a_cosine_with_a_one_step_budget_does_not_divide_by_zero() -> None:
    schedule = LearningRateSchedule(decay="cosine")
    assert schedule.multiplier(0, 1) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"warmup_steps": -1},
        {"decay": "linear"},
        {"min_lr_fraction": 1.5},
        {"min_lr_fraction": float("nan")},
    ],
)
def test_bad_settings_are_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LearningRateSchedule(**bad)  # type: ignore[arg-type]
