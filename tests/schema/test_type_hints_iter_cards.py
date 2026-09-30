from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.type_hints import (
    InputShape,
    batched_input_shape_of,
    iter_cards,
    map_cards,
)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(DataSource.SCRYFALL, name, datetime.now(timezone.utc)),
    )


def test_iter_cards_flattens_every_shape_in_order_keeping_duplicates() -> None:
    a, b = _card("a"), _card("b")

    assert list(iter_cards(a)) == [a]
    assert list(iter_cards([a, b])) == [a, b]
    assert list(iter_cards([[a], [b, a]])) == [a, b, a]  # type: ignore[arg-type]
    assert list(iter_cards([[[a]], [[b], [a]]])) == [a, b, a]  # type: ignore[arg-type]
    assert list(iter_cards([])) == []


def test_batched_input_shape_of_reads_each_depth_as_a_batch() -> None:
    a, b = _card("a"), _card("b")

    cases: list[tuple[Any, InputShape]] = [
        ([a, b], InputShape.SINGLE_CARD),
        ([[a, b], [a]], InputShape.MULTI_CARD),
        ([[[a]], [[b]]], InputShape.MULTI_GROUP),
        ([[[[a]]]], InputShape.BATCHED_MULTI_GROUP),  # one level too deep
    ]
    for batch, expected in cases:
        assert batched_input_shape_of(batch) is expected


def test_batched_input_shape_of_rejects_non_batches() -> None:
    import pytest

    with pytest.raises(TypeError):
        batched_input_shape_of(_card("a"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        batched_input_shape_of([])
    with pytest.raises(ValueError):
        batched_input_shape_of([[]])  # type: ignore[arg-type]


def test_map_cards_preserves_shape_and_order() -> None:
    a, b, c = _card("A"), _card("B"), _card("C")
    names: list[str] = []

    def record(card: GenericCard) -> GenericCard:
        names.append(card.name)
        return card

    grouped = [[a], [b, c]]
    result = map_cards(grouped, record)

    assert result == grouped and result is not grouped
    assert result[0] is not grouped[0]
    assert names == ["A", "B", "C"]
    assert map_cards(a, record) is a
