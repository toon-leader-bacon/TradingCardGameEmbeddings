from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.pool_mirror import (
    delete_cards_except,
    store_mirrored_card,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_GAME = GameId.YUGIOH


def _card(name: str, text: str = "t") -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=_GAME,
        name=name,
        raw_content={"name": name, "text": text},
        provenance=Provenance(
            data_source=DataSource.YGOPRODECK,
            source_id="1",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def test_store_creates_a_new_card() -> None:
    binder = CardBinder.load([])
    card = _card("A")
    assert store_mirrored_card(binder, None, card) == (card.nocab_uuid, True)
    assert binder.get_by_uuid(card.nocab_uuid) is not None


def test_store_merges_onto_existing_only_when_content_differs() -> None:
    binder = CardBinder.load([])
    existing = _card("A")
    binder.create(existing)

    assert store_mirrored_card(binder, existing, _card("A")) == (
        existing.nocab_uuid,
        False,
    )
    assert store_mirrored_card(binder, existing, _card("A", "new")) == (
        existing.nocab_uuid,
        True,
    )
    stored = binder.get_by_uuid(existing.nocab_uuid)
    assert stored is not None and stored.raw_content["text"] == "new"


def test_delete_keeps_the_kept_cards_and_the_unknown_sentinel() -> None:
    binder = CardBinder.load([])
    kept, dropped = _card("Kept"), _card("Dropped")
    binder.create(kept)
    binder.create(dropped)
    binder.ensure_unknown_card(_GAME)

    delete_cards_except(binder, _GAME, {kept.nocab_uuid})

    assert set(binder.all_uuids(_GAME)) == {
        kept.nocab_uuid,
        CardBinder.unknown_card_uuid(_GAME),
    }
