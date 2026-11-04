"""RoundSchedule and draw_round_schedule: the same draws as step by step,
and the schedule check every feed makes."""

import random
from typing import Sequence

import pytest

from src.dojos.dojo import Dojo
from src.training.diet.batch_feed_errors import ScheduleMismatchError
from src.training.diet.diet_sampler import TemperatureDietSampler
from src.training.diet.round_schedule import (
    RoundSchedule,
    draw_round_schedule,
    require_next_step,
)
from tests.training.fakes import FakeDojo


def _dojos(*names: str) -> Sequence[Dojo]:
    return [FakeDojo(name) for name in names]


def test_the_schedule_is_the_draws_the_trainer_would_make_step_by_step() -> None:
    active = _dojos("a", "b", "c")
    sampler = TemperatureDietSampler(0.0)
    stepwise_rng = random.Random(7)
    stepwise = [sampler.next_dojo(active, stepwise_rng).name for _ in range(20)]
    rng = random.Random(7)
    assert draw_round_schedule(sampler, active, 20, rng).dojo_names == tuple(stepwise)
    assert rng.getstate() == stepwise_rng.getstate()


def test_drawing_needs_dojos_and_steps() -> None:
    sampler = TemperatureDietSampler(1.0)
    with pytest.raises(ValueError, match="active dojo"):
        draw_round_schedule(sampler, [], 3, random.Random(0))
    with pytest.raises(ValueError, match="at least one step"):
        draw_round_schedule(sampler, _dojos("a"), 0, random.Random(0))


def test_a_schedule_is_never_empty() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        RoundSchedule(())


def test_require_next_step_accepts_only_the_next_scheduled_dojo() -> None:
    require_next_step(["a", "b"], "a")
    with pytest.raises(ScheduleMismatchError, match="no round open"):
        require_next_step(None, "a")
    with pytest.raises(ScheduleMismatchError, match="past the round"):
        require_next_step([], "a")
    with pytest.raises(ScheduleMismatchError, match="next scheduled dojo is 'a'"):
        require_next_step(["a"], "b")
