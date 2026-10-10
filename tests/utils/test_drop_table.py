import math
import random
from collections import Counter

import pytest

from src.utils.drop_table import DropTable, DropTableEntry


class TestConstruction:
    @pytest.mark.parametrize("weight", [-1.0, math.nan, math.inf])
    def test_rejects_bad_weights(self, weight: float) -> None:
        with pytest.raises(ValueError):
            DropTableEntry(weight, "a")

    def test_rejects_a_table_with_nothing_pullable(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DropTable.of([(0, "a"), (0, "b")])

    def test_rejects_an_empty_table(self) -> None:
        with pytest.raises(ValueError):
            DropTable.of([])

    def test_rejects_entries_that_are_not_a_tuple(self) -> None:
        with pytest.raises(TypeError):
            DropTable([DropTableEntry(1, "a")])  # type: ignore[arg-type]

    def test_uniform_needs_an_outcome(self) -> None:
        with pytest.raises(ValueError):
            DropTable.uniform([])

    def test_total_weight_counts_only_this_level(self) -> None:
        sub = DropTable.of([(5, "x"), (5, "y")])
        assert DropTable.of([(1, "a"), (2, sub)]).total_weight == 3


class TestPull:
    def test_a_single_outcome_always_comes_up(self) -> None:
        table = DropTable.of([(1, "a"), (0, "b")])
        rng = random.Random(0)
        assert {table.pull(rng) for _ in range(50)} == {"a"}

    def test_frequencies_follow_the_weights(self) -> None:
        table = DropTable.of([(3, "a"), (1, "b")])
        rng = random.Random(0)
        counts = Counter(table.pull(rng) for _ in range(20_000))
        assert counts["a"] / 20_000 == pytest.approx(0.75, abs=0.02)

    def test_rolls_into_sub_tables(self) -> None:
        meta = DropTable.of([(1, "power"), (1, "armor")])
        table = DropTable.of([(1, "nothing"), (1, meta)])
        rng = random.Random(0)
        counts = Counter(table.pull(rng) for _ in range(20_000))
        assert set(counts) == {"nothing", "power", "armor"}
        assert counts["power"] / 20_000 == pytest.approx(0.25, abs=0.02)

    def test_is_reproducible_with_the_same_seed(self) -> None:
        table = DropTable.uniform(range(10))
        first = [table.pull(random.Random(7)) for _ in range(5)]
        second = [table.pull(random.Random(7)) for _ in range(5)]
        assert first == second

    def test_none_is_a_legitimate_outcome(self) -> None:
        assert DropTable.of([(1, None)]).pull(random.Random(0)) is None

    def test_rounding_past_the_end_returns_the_last_positive_entry(self) -> None:
        table = DropTable.of([(1, "a"), (1, "b"), (0, "c")])
        assert table._entry_at(table.total_weight).outcome == "b"


class TestFiltered:
    def test_keeps_only_accepted_outcomes(self) -> None:
        table = DropTable.uniform([1, 2, 3, 4]).filtered(lambda n: n % 2 == 0)
        assert table is not None
        assert [entry.outcome for entry in table.entries] == [2, 4]

    def test_returns_none_when_nothing_survives(self) -> None:
        assert DropTable.uniform([1, 3]).filtered(lambda n: n % 2 == 0) is None

    def test_returns_none_when_only_zero_weights_survive(self) -> None:
        table = DropTable.of([(0, "keep"), (1, "drop")])
        assert table.filtered(lambda outcome: outcome == "keep") is None

    def test_drops_emptied_sub_tables_and_filters_kept_ones(self) -> None:
        empty_after = DropTable.of([(1, "x")])
        kept = DropTable.of([(1, "y"), (1, "z")])
        table = DropTable.of([(1, "a"), (2, empty_after), (3, kept)])

        filtered = table.filtered(lambda outcome: outcome in {"a", "y"})

        assert filtered is not None
        assert [entry.weight for entry in filtered.entries] == [1, 3]
        sub = filtered.entries[1].outcome
        assert isinstance(sub, DropTable)
        assert [entry.outcome for entry in sub.entries] == ["y"]

    def test_leaves_the_original_unchanged(self) -> None:
        table = DropTable.uniform(["a", "b"])
        table.filtered(lambda outcome: outcome == "a")
        assert len(table.entries) == 2

    def test_keeps_a_none_outcome_the_predicate_accepts(self) -> None:
        filtered = DropTable.of([(1, None), (1, "b")]).filtered(lambda o: o is None)
        assert filtered is not None
        assert filtered.pull(random.Random(0)) is None


class TestOutcomeProbabilities:
    def test_nested_tables_multiply_down(self) -> None:
        table = DropTable.of([(3, "a"), (1, DropTable.uniform(["b", "c"]))])
        assert table.outcome_probabilities() == pytest.approx(
            {"a": 0.75, "b": 0.125, "c": 0.125}
        )

    def test_zero_weight_outcomes_are_absent(self) -> None:
        assert DropTable.of([(1, "a"), (0, "b")]).outcome_probabilities() == {"a": 1.0}

    def test_a_repeated_outcome_sums_its_rows(self) -> None:
        table = DropTable.of([(1, "a"), (1, DropTable.of([(1, "a"), (1, "b")]))])
        assert table.outcome_probabilities() == pytest.approx({"a": 0.75, "b": 0.25})

    def test_matches_pull_frequencies(self) -> None:
        table = DropTable.of([(1, "a"), (3, DropTable.of([(1, "b"), (2, "c")]))])
        rng = random.Random(0)
        counts = Counter(table.pull(rng) for _ in range(6000))
        for outcome, probability in table.outcome_probabilities().items():
            assert abs(counts[outcome] / 6000 - probability) < 0.03
