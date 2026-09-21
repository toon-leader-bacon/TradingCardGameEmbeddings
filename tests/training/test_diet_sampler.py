import random
from collections import Counter

import pytest

from src.dojos.dojo import Dojo
from src.training.diet.diet_sampler import (
    DietSampler,
    TemperatureDietSampler,
    diet_sampler_for,
)
from src.training.plan import Proportional, Temperature, Uniform
from tests.training.fakes import FakeDojo


def _draw(
    sampler: DietSampler, dojos: list[FakeDojo], rng: random.Random
) -> Counter[str]:
    active: list[Dojo] = list(dojos)
    return Counter(sampler.next_dojo(active, rng).name for _ in range(500))


def _draw_counts(alpha: float, dojos: list[FakeDojo], n: int = 2000) -> Counter[str]:
    sampler = TemperatureDietSampler(alpha)
    rng = random.Random(0)
    return Counter(sampler.next_dojo(dojos, rng).name for _ in range(n))  # type: ignore[arg-type]


def test_uniform_ignores_example_counts() -> None:
    counts = _draw_counts(
        0.0, [FakeDojo("a", train_count=1), FakeDojo("b", train_count=1000)]
    )
    assert abs(counts["a"] - counts["b"]) < 300


def test_proportional_follows_example_counts() -> None:
    counts = _draw_counts(
        1.0, [FakeDojo("a", train_count=1), FakeDojo("b", train_count=9)]
    )
    assert counts["b"] > 5 * counts["a"]


def test_a_dojo_with_no_train_examples_is_never_drawn() -> None:
    counts = _draw_counts(0.0, [FakeDojo("a", train_count=0), FakeDojo("b")])
    assert "a" not in counts


def test_raises_on_empty_or_exampleless_active_set() -> None:
    sampler = TemperatureDietSampler(1.0)
    with pytest.raises(ValueError):
        sampler.next_dojo([], random.Random(0))
    with pytest.raises(ValueError):
        sampler.next_dojo([FakeDojo("a", train_count=0)], random.Random(0))  # type: ignore[list-item]  # noqa: E501


def test_factory_maps_each_rule_to_its_alpha() -> None:
    dojos = [FakeDojo("a", train_count=1), FakeDojo("b", train_count=9)]
    rng = random.Random(0)
    uniform = _draw(diet_sampler_for(Uniform()), dojos, rng)
    proportional = _draw(diet_sampler_for(Proportional()), dojos, rng)
    tempered = diet_sampler_for(Temperature(0.5))
    assert uniform["a"] > proportional["a"]
    assert _draw(tempered, dojos, rng).total() == 500
