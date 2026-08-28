from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.name_lookup import find_uuid_by_name
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str, source_game: GameId = GameId.MTG) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=source_game,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestFindUuidByName:
    def test_exact_match_resolves(self) -> None:
        registry = CardBinder()
        card = _card("Bolt")
        registry.add(card)

        assert find_uuid_by_name(registry, GameId.MTG, "Bolt") == card.nocab_uuid

    def test_mdfc_fallback_resolves_single_match(self) -> None:
        registry = CardBinder()
        card = _card("Bruce Banner // The Incredible Hulk")
        registry.add(card)

        assert (
            find_uuid_by_name(registry, GameId.MTG, "Bruce Banner") == card.nocab_uuid
        )

    def test_no_match_at_all_is_unresolved(self) -> None:
        registry = CardBinder()

        assert find_uuid_by_name(registry, GameId.MTG, "Nonexistent") is None

    def test_ambiguous_fallback_match_is_unresolved(self) -> None:
        registry = CardBinder()
        registry.add(_card("Bolt // Bolt Alpha"))
        registry.add(_card("Bolt // Bolt Beta"))

        assert find_uuid_by_name(registry, GameId.MTG, "Bolt") is None
