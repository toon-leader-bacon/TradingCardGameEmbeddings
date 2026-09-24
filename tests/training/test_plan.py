import pytest

from src.training.plan import FaultPolicy, HardwareLimits, Temperature
from tests.training.fakes import make_phase, make_plan, saturation_spec


def test_temperature_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError):
        Temperature(alpha=-1)


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_saturation_spec_rejects_fraction_outside_unit_interval(
    fraction: float,
) -> None:
    with pytest.raises(ValueError):
        saturation_spec(target_saturated_fraction=fraction)


def test_saturation_spec_rejects_zero_patience() -> None:
    with pytest.raises(ValueError):
        saturation_spec(patience_rounds=0)


@pytest.mark.parametrize("field", ["steps_per_round", "max_rounds"])
def test_phase_rejects_non_positive_counts(field: str) -> None:
    with pytest.raises(ValueError):
        make_phase(("a",), **{field: 0})


def test_phase_needs_a_dojo() -> None:
    with pytest.raises(ValueError):
        make_phase(())


def test_plan_needs_a_phase() -> None:
    with pytest.raises(ValueError):
        make_plan(())


def test_plan_rejects_a_held_out_dojo_in_a_diet() -> None:
    with pytest.raises(ValueError):
        make_plan((make_phase(("a", "b")),), held_out_dojos=frozenset({"b"}))


def test_limits_and_fault_policy_reject_non_positive_values() -> None:
    with pytest.raises(ValueError):
        HardwareLimits(max_batch_cost=0)
    with pytest.raises(ValueError):
        HardwareLimits(max_batch_cost=1, precision="fp8")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FaultPolicy(max_consecutive_dojo_failures=0)


def test_phase_rejects_bad_learning_rates_and_grad_norm() -> None:
    bad = ({"encoder_lr": -1.0}, {"head_lr": float("nan")}, {"max_grad_norm": 0.0})
    for overrides in bad:
        with pytest.raises(ValueError):
            make_phase(("a",), **overrides)


def test_phase_rejects_duplicate_dojos_and_unsafe_names() -> None:
    with pytest.raises(ValueError):
        make_phase(("a", "a"))
    with pytest.raises(ValueError):
        make_phase(("a",), name="bad name/1")


def test_plan_rejects_duplicate_phase_names_and_no_eval_examples() -> None:
    with pytest.raises(ValueError):
        make_plan((make_phase(("a",)), make_phase(("a",))))
    with pytest.raises(ValueError):
        make_plan((make_phase(("a",)),), eval_examples_per_dojo=0)


def test_plan_rejects_phase_names_differing_only_by_case() -> None:
    with pytest.raises(ValueError):
        make_plan((make_phase(("a",), name="A"), make_phase(("a",), name="a")))
