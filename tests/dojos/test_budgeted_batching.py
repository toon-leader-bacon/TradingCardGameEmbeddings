import random
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.dojos.budgeted_batching import group_by_budget, padded_cost
from src.dojos.dojo import BatchBudget
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.type_hints import TrainingDatum


def _card() -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="c",
        raw_content={},
        provenance=Provenance(DataSource.SCRYFALL, "c", datetime.now(timezone.utc)),
    )


def _datum(card_count: int) -> TrainingDatum[float]:
    return ([_card() for _ in range(card_count)], 0.0)


def test_packs_examples_up_to_the_budget() -> None:
    data = [_datum(2) for _ in range(5)]

    groups = list(group_by_budget(data, BatchBudget(4, lambda card: 1)))

    assert [len(g) for g in groups] == [2, 2, 1]


def test_respects_a_non_unit_card_cost() -> None:
    data = [_datum(1) for _ in range(4)]

    groups = list(group_by_budget(data, BatchBudget(4, lambda card: 2)))

    assert [len(g) for g in groups] == [2, 2]


def test_raises_when_one_example_exceeds_the_budget() -> None:
    with pytest.raises(ValueError):
        list(group_by_budget([_datum(5)], BatchBudget(4, lambda card: 1)))


def test_empty_input_yields_nothing() -> None:
    assert list(group_by_budget([], BatchBudget(4, lambda card: 1))) == []


def test_padded_cost_charges_every_example_the_largest_cost() -> None:
    assert padded_cost([60, 600, 60]) == 1800
    assert padded_cost([3]) == 3
    assert padded_cost([]) == 0


def test_groups_are_charged_their_padded_cost() -> None:
    # Costs 2, 2, 1, 5, 1 under budget 6: 3 x 2 = 6 fits; the 5 would make
    # it 4 x 5, so it starts a group; 2 x 5 > 6, so the last 1 starts one
    data = [_datum(n) for n in (2, 2, 1, 5, 1)]

    groups = list(group_by_budget(data, BatchBudget(6, lambda card: 1)))

    assert [len(g) for g in groups] == [3, 1, 1]


def test_an_outlier_first_keeps_its_group_small() -> None:
    # A 600-card deck makes every deck in its group cost 600
    data = [_datum(600)] + [_datum(60) for _ in range(20)]

    groups = list(group_by_budget(data, BatchBudget(1200, lambda card: 1)))

    assert [len(g) for g in groups] == [2, 19]


@pytest.mark.parametrize("seed", range(5))
def test_no_group_exceeds_the_budget_in_any_order(seed: int) -> None:
    sizes = [60] * 30 + [600, 300, 100]
    random.Random(seed).shuffle(sizes)
    data = [_datum(n) for n in sizes]

    groups = list(group_by_budget(data, BatchBudget(1200, lambda card: 1)))

    assert sum(len(g) for g in groups) == len(sizes)
    for group in groups:
        assert padded_cost([len(datum[0]) for datum in group]) <= 1200
