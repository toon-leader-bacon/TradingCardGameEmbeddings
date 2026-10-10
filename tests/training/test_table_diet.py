import logging
import random
from collections import Counter
from typing import Sequence

import pytest

from src.dojos.dojo import Dojo
from src.training.diet.diet_sampler import diet_sampler_for
from src.training.diet.table_diet import (
    EmptiedRow,
    TableDietSampler,
    as_table_diet,
    build_dojo_drop_table,
)
from src.training.plan import (
    DojoGroupRow,
    DojoRow,
    Proportional,
    SubTableRow,
    TableDiet,
    Uniform,
    leaf_dojo_names,
)
from tests.training.fakes import FakeDojo

# 50/50 between a contrastive pair and a metric group
_HALVES = TableDiet(
    (
        SubTableRow(1, (DojoRow(1, "c1"), DojoRow(1, "c2"))),
        DojoGroupRow(1, ("m1", "m2"), Proportional()),
    )
)


def _dojos(*names: str, train_count: int = 100) -> list[Dojo]:
    return [FakeDojo(name, train_count=train_count) for name in names]  # type: ignore[misc]


def _draw(
    sampler: TableDietSampler, active: Sequence[Dojo], n: int = 4000
) -> Counter[str]:
    rng = random.Random(0)
    return Counter(sampler.next_dojo(active, rng).name for _ in range(n))


class TestTableDiet:
    def test_dojo_names_are_the_leaves_in_row_order(self) -> None:
        assert _HALVES.dojo_names == ("c1", "c2", "m1", "m2")

    def test_leaf_dojo_names_walks_nested_tables(self) -> None:
        rows = (DojoRow(1, "a"), SubTableRow(1, (DojoRow(1, "b"),)))
        assert leaf_dojo_names(rows) == ("a", "b")

    def test_a_dojo_under_two_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            TableDiet((DojoRow(1, "a"), DojoGroupRow(1, ("a", "b"), Uniform())))

    @pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
    def test_bad_row_weights_raise(self, weight: float) -> None:
        with pytest.raises(ValueError, match="weight"):
            DojoRow(weight, "a")

    def test_an_all_zero_table_raises(self) -> None:
        with pytest.raises(ValueError, match="positive weight"):
            TableDiet((DojoRow(0, "a"),))

    def test_empty_tables_and_groups_raise(self) -> None:
        with pytest.raises(ValueError):
            TableDiet(())
        with pytest.raises(ValueError):
            SubTableRow(1, ())
        with pytest.raises(ValueError):
            DojoGroupRow(1, (), Uniform())


class TestBuildDojoDropTable:
    def test_shares_follow_the_nested_weights(self) -> None:
        counts = {"c1": 5, "c2": 5, "m1": 1, "m2": 3}
        built = build_dojo_drop_table(_HALVES, counts)
        assert built.table is not None
        assert built.table.outcome_probabilities() == pytest.approx(
            {"c1": 0.25, "c2": 0.25, "m1": 0.125, "m2": 0.375}
        )
        assert built.emptied_rows == ()

    def test_a_missing_dojo_gives_its_share_to_its_siblings(self) -> None:
        built = build_dojo_drop_table(_HALVES, {"c1": 5, "m1": 1, "m2": 1})
        assert built.table is not None
        assert built.table.outcome_probabilities()["c1"] == pytest.approx(0.5)

    def test_an_emptied_row_drops_out_and_is_named(self) -> None:
        built = build_dojo_drop_table(_HALVES, {"m1": 1, "m2": 1})
        assert built.table is not None
        assert built.table.outcome_probabilities() == pytest.approx(
            {"m1": 0.5, "m2": 0.5}
        )
        assert built.emptied_rows == (EmptiedRow("table[0]", ("c1", "c2")),)

    def test_nested_emptied_rows_carry_their_path(self) -> None:
        diet = TableDiet(
            (
                DojoRow(1, "a"),
                SubTableRow(1, (DojoGroupRow(1, ("b",), Uniform()),)),
            )
        )
        built = build_dojo_drop_table(diet, {"a": 1})
        assert [row.path for row in built.emptied_rows] == [
            "table[1].table[0]",
            "table[1]",
        ]
        assert str(built.emptied_rows[0]) == "table[1].table[0] (dojos: b)"

    def test_zero_counts_are_not_drawable(self) -> None:
        built = build_dojo_drop_table(_HALVES, {"c1": 0, "c2": 0, "m1": 2, "m2": 0})
        assert built.table is not None
        assert built.table.outcome_probabilities() == {"m1": 1.0}

    def test_only_zero_weight_survivors_is_empty(self) -> None:
        diet = TableDiet(
            (DojoRow(1, "a"), SubTableRow(1, (DojoRow(1, "b"), DojoRow(0, "c"))))
        )
        built = build_dojo_drop_table(diet, {"a": 1, "c": 1})
        assert [row.path for row in built.emptied_rows] == ["table[1]"]

    def test_nothing_drawable_is_none(self) -> None:
        built = build_dojo_drop_table(_HALVES, {})
        assert built.table is None
        assert len(built.emptied_rows) == 2


class TestAsTableDiet:
    def test_a_table_is_itself(self) -> None:
        assert as_table_diet(_HALVES, ["x"]) is _HALVES

    def test_a_flat_rule_is_one_group_row(self) -> None:
        assert as_table_diet(Uniform(), ["a"]).rows == (
            DojoGroupRow(1.0, ("a",), Uniform()),
        )


class TestTableDietSampler:
    def test_draws_follow_the_table(self) -> None:
        counts = _draw(TableDietSampler(_HALVES), _dojos("c1", "c2", "m1", "m2"))
        contrastive = counts["c1"] + counts["c2"]
        assert abs(contrastive - 2000) < 200

    def test_a_saturated_dojos_share_stays_in_its_half(self) -> None:
        counts = _draw(TableDietSampler(_HALVES), _dojos("c1", "m1", "m2"))
        assert abs(counts["c1"] - 2000) < 200

    def test_an_emptied_row_is_logged_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sampler = TableDietSampler(_HALVES)
        rng = random.Random(0)
        with caplog.at_level(logging.WARNING):
            sampler.next_dojo(_dojos("m1", "m2"), rng)
            sampler.next_dojo(_dojos("m1"), rng)
            sampler.next_dojo(_dojos("m1", "m2"), rng)
        emptied = [r for r in caplog.records if "table[0]" in r.getMessage()]
        assert len(emptied) == 1

    def test_raises_on_empty_strange_or_undrawable_active_sets(self) -> None:
        sampler = TableDietSampler(_HALVES)
        rng = random.Random(0)
        with pytest.raises(ValueError, match="no active"):
            sampler.next_dojo([], rng)
        with pytest.raises(ValueError, match="not in the diet table"):
            sampler.next_dojo(_dojos("zzz"), rng)
        with pytest.raises(ValueError, match="no active dojo is drawable"):
            sampler.next_dojo(_dojos("c1", train_count=0), rng)

    def test_the_factory_builds_it_for_a_table_diet(self) -> None:
        assert isinstance(diet_sampler_for(_HALVES), TableDietSampler)
