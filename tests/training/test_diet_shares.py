import csv
from pathlib import Path

import pytest

from src.dojos.dojo import Dojo
from src.schema.holdout import HoldoutSpec
from src.training.diet.diet_shares import (
    DietShare,
    expected_dojo_shares,
    format_diet_shares,
    plan_diet_shares,
    write_diet_shares_csv,
)
from src.training.plan import (
    DojoGroupRow,
    DojoRow,
    Phase,
    SaturationSpec,
    TableDiet,
    Temperature,
    TrainingPlan,
    Uniform,
)
from tests.training.fakes import FakeDojo


def _phase(name: str, dojo_names: tuple[str, ...], diet: object) -> Phase:
    return Phase(
        name=name,
        dojo_names=dojo_names,
        diet_rule=diet,  # type: ignore[arg-type]
        encoder_trainable=False,
        encoder_lr=0.0,
        head_lr=1e-3,
        steps_per_round=1,
        max_rounds=1,
        saturation=SaturationSpec(0.0, 1, 0.1, 1.0),
    )


class TestExpectedDojoShares:
    def test_uniform_ignores_counts(self) -> None:
        shares = expected_dojo_shares(Uniform(), ["a", "b"], {"a": 10, "b": 99})
        assert shares == {"a": 0.5, "b": 0.5}

    def test_temperature_weighs_counts(self) -> None:
        shares = expected_dojo_shares(Temperature(0.5), ["a", "b"], {"a": 1, "b": 9})
        assert shares == pytest.approx({"a": 0.25, "b": 0.75})

    def test_undrawn_dojos_get_zero_in_dojo_order(self) -> None:
        diet = TableDiet((DojoRow(1, "a"), DojoRow(0, "b"), DojoRow(1, "c")))
        shares = expected_dojo_shares(diet, ["a", "b", "c"], {"a": 1, "b": 1})
        assert list(shares) == ["a", "b", "c"]
        assert shares == {"a": 1.0, "b": 0.0, "c": 0.0}

    def test_nothing_drawable_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_dojo_shares(Uniform(), ["a"], {"a": 0})


class TestPlanDietShares:
    def test_every_phase_in_order(self) -> None:
        dojos: list[Dojo] = [
            FakeDojo("a", train_count=10),  # type: ignore[list-item]
            FakeDojo("b", train_count=30),  # type: ignore[list-item]
        ]
        diet = TableDiet(
            (DojoRow(3, "a"), DojoGroupRow(1, ("b",), Uniform())),
        )
        plan = TrainingPlan(
            phases=(
                _phase("one", ("a", "b"), Uniform()),
                _phase("two", ("a", "b"), diet),
            ),
            holdout=HoldoutSpec.no_holdout(),
            held_out_dojos=frozenset(),
            eval_examples_per_dojo=1,
            seed=0,
        )
        shares = plan_diet_shares(plan, dojos)
        assert shares == [
            DietShare("one", "a", 10, 0.5),
            DietShare("one", "b", 30, 0.5),
            DietShare("two", "a", 10, 0.75),
            DietShare("two", "b", 30, 0.25),
        ]


class TestFormatAndWrite:
    def test_format(self) -> None:
        lines = format_diet_shares([DietShare("frozen", "a", 10, 0.5)])
        assert lines == ["[frozen]  50.00%  a (train 10)"]

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "diet_shares.csv"
        write_diet_shares_csv([DietShare("frozen", "a", 10, 0.5)], path)
        with path.open(encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        assert rows == [
            ["phase", "dojo", "train_count", "share"],
            ["frozen", "a", "10", "0.5"],
        ]
