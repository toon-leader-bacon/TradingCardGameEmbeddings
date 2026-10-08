from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import uuid_for_name_or_front_face
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _add_card(binder: CardBinder, name: str) -> GenericCard:
    card = GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    binder.create(card)
    return card


class TestUuidForNameOrFrontFace:
    def test_exact_name(self) -> None:
        binder = CardBinder()
        card = _add_card(binder, "Lightning Bolt")

        found = uuid_for_name_or_front_face(binder, GameId.MTG, "Lightning Bolt")

        assert found == card.nocab_uuid

    def test_front_face(self) -> None:
        binder = CardBinder()
        card = _add_card(binder, "Hengegate Pathway // Mistgate Pathway")

        found = uuid_for_name_or_front_face(binder, GameId.MTG, "Hengegate Pathway")

        assert found == card.nocab_uuid

    def test_back_face(self) -> None:
        binder = CardBinder()
        card = _add_card(binder, "Hengegate Pathway // Mistgate Pathway")

        found = uuid_for_name_or_front_face(binder, GameId.MTG, "Mistgate Pathway")

        assert found == card.nocab_uuid

    def test_a_face_shared_by_two_cards_is_not_guessed(self) -> None:
        binder = CardBinder()
        _add_card(binder, "Alpha // Shared")
        _add_card(binder, "Beta // Shared")

        assert uuid_for_name_or_front_face(binder, GameId.MTG, "Shared") is None

    def test_a_partial_name_is_not_a_face(self) -> None:
        binder = CardBinder()
        _add_card(binder, "Hengegate Pathway // Mistgate Pathway")

        assert uuid_for_name_or_front_face(binder, GameId.MTG, "gate Pathway") is None

    def test_registered_alternate_name(self) -> None:
        binder = CardBinder()
        card = _add_card(binder, "Masked Meower")
        binder.register_alias(
            GameId.MTG, DataSource.PRINTED_NAME, "Skittering Kitten", card.nocab_uuid
        )

        found = uuid_for_name_or_front_face(binder, GameId.MTG, "Skittering Kitten")

        assert found == card.nocab_uuid

    def test_unknown_name(self) -> None:
        binder = CardBinder()
        _add_card(binder, "Lightning Bolt")

        assert uuid_for_name_or_front_face(binder, GameId.MTG, "Nothing") is None
