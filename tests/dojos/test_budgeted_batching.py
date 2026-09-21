from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.dojos.budgeted_batching import group_by_budget
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


def _datum(card_count: int) -> TrainingDatum:
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
