from uuid import uuid4

from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards


class TestDeckUuidFromCards:
    def test_deterministic_across_calls(self) -> None:
        cards = [uuid4(), uuid4(), uuid4()]

        assert deck_uuid_from_cards(cards) == deck_uuid_from_cards(cards)

    def test_order_independent(self) -> None:
        cards = [uuid4(), uuid4(), uuid4()]

        assert deck_uuid_from_cards(cards) == deck_uuid_from_cards(
            list(reversed(cards))
        )

    def test_different_multisets_hash_differently(self) -> None:
        assert deck_uuid_from_cards([uuid4()]) != deck_uuid_from_cards([uuid4()])

    def test_copy_count_matters(self) -> None:
        card = uuid4()

        assert deck_uuid_from_cards([card]) != deck_uuid_from_cards([card, card])

    def test_empty_deck_is_deterministic(self) -> None:
        assert deck_uuid_from_cards([]) == deck_uuid_from_cards([])
