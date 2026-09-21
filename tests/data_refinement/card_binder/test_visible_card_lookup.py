from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.visible_card_lookup import VisibleCardLookup
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from src.schema.splits import Split


def _card(name: str, game: GameId = GameId.MTG) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder_with_one_card_per_tier() -> tuple[CardBinder, HoldoutSpec, dict]:
    spec = HoldoutSpec(3, (4, 3, 3), frozenset())
    binder = CardBinder()
    by_tier: dict[CardTier, GenericCard] = {}
    i = 0
    while len(by_tier) < 3:
        card = _card(f"card-{i}")
        i += 1
        tier = spec.tier_of(card.nocab_uuid, card.source_game)
        if tier not in by_tier:
            by_tier[tier] = binder.create(card)
    return binder, spec, by_tier


def test_train_view_sees_only_train_tier() -> None:
    binder, spec, by_tier = _binder_with_one_card_per_tier()
    view = VisibleCardLookup(binder, spec, Split.TRAIN)
    assert view.get_by_uuid(by_tier[CardTier.TRAIN].nocab_uuid) is not None
    assert view.get_by_uuid(by_tier[CardTier.TEST].nocab_uuid) is None
    assert view.get_by_uuid(by_tier[CardTier.VALIDATION].nocab_uuid) is None


def test_validation_view_sees_everything() -> None:
    binder, spec, by_tier = _binder_with_one_card_per_tier()
    view = VisibleCardLookup(binder, spec, Split.VALIDATION)
    assert {c.nocab_uuid for c in view.all_cards(GameId.MTG)} == {
        c.nocab_uuid for c in by_tier.values()
    }


def test_name_and_enumeration_methods_filter() -> None:
    binder, spec, by_tier = _binder_with_one_card_per_tier()
    view = VisibleCardLookup(binder, spec, Split.TEST)
    hidden = by_tier[CardTier.VALIDATION]
    assert view.get_by_name(GameId.MTG, hidden.name) == []
    assert view.get_by_name_single(GameId.MTG, hidden.name) is None
    assert view.get_by_name_regex(GameId.MTG, hidden.name) == []
    assert hidden.nocab_uuid not in set(view.all_uuids())
    assert by_tier[CardTier.TEST].nocab_uuid in set(view.all_uuids(GameId.MTG))
