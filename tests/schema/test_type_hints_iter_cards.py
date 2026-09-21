from datetime import datetime, timezone
from uuid import uuid4

from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.type_hints import iter_cards


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
