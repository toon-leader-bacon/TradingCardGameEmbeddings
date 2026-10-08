from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors.row_values import (
    option_cards_and_pick_index,
    option_cards_for_uuids,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder(*cards: GenericCard) -> CardBinder:
    binder = CardBinder()
    for card in cards:
        binder.create(card)
    return binder


class TestOptionCardsForUuids:
    def test_returns_the_cards_in_the_cells_order(self) -> None:
        a, b = _card("A"), _card("B")

        result = option_cards_for_uuids(
            _binder(a, b), [str(b.nocab_uuid), str(a.nocab_uuid)]
        )

        assert result == [b, a]

    def test_an_unparseable_uuid_voids_the_row(self) -> None:
        a = _card("A")

        assert option_cards_for_uuids(_binder(a), [str(a.nocab_uuid), "nope"]) is None

    def test_a_uuid_the_lookup_lacks_voids_the_row(self) -> None:
        a, stranger = _card("A"), _card("Stranger")

        result = option_cards_for_uuids(
            _binder(a), [str(a.nocab_uuid), str(stranger.nocab_uuid)]
        )

        assert result is None

    def test_an_empty_list_is_an_empty_list(self) -> None:
        assert option_cards_for_uuids(_binder(), []) == []


class TestOptionCardsAndPickIndexStillBuildsOnIt:
    def test_returns_the_cards_and_the_picked_position(self) -> None:
        a, b = _card("A"), _card("B")

        result = option_cards_and_pick_index(
            _binder(a, b), [str(a.nocab_uuid), str(b.nocab_uuid)], str(b.nocab_uuid)
        )

        assert result == ([a, b], 1)

    def test_a_null_pick_is_still_dropped(self) -> None:
        a = _card("A")

        assert (
            option_cards_and_pick_index(_binder(a), [str(a.nocab_uuid)], None) is None
        )
