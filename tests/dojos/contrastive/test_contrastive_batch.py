from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str = "card") -> GenericCard:
    return GenericCard(
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


class TestPostInit:
    def test_accepts_a_well_formed_batch(self) -> None:
        cards = [_card(), _card(), _card()]
        identities = [(uuid4(),) for _ in cards]

        batch = ContrastiveBatch(
            items=cards, identities=identities, positive_cliques=[[0, 1]]
        )

        assert batch.items == cards

    def test_accepts_an_empty_batch(self) -> None:
        batch = ContrastiveBatch(items=[], identities=[], positive_cliques=[])

        assert batch.items == []

    def test_raises_on_items_identities_length_mismatch(self) -> None:
        cards = [_card(), _card()]

        with pytest.raises(ValueError):
            ContrastiveBatch(items=cards, identities=[(uuid4(),)], positive_cliques=[])

    def test_raises_on_mixed_input_shapes(self) -> None:
        single_card = _card()
        multi_card = [_card(), _card()]

        with pytest.raises(ValueError):
            ContrastiveBatch(
                items=[single_card, multi_card],  # type: ignore[list-item]
                identities=[(uuid4(),), (uuid4(),)],
                positive_cliques=[],
            )

    def test_raises_on_out_of_range_positive_group_index(self) -> None:
        cards = [_card(), _card()]
        identities = [(uuid4(),) for _ in cards]

        with pytest.raises(ValueError):
            ContrastiveBatch(
                items=cards, identities=identities, positive_cliques=[[0, 5]]
            )

    def test_raises_on_duplicate_index_within_a_group(self) -> None:
        cards = [_card(), _card()]
        identities = [(uuid4(),) for _ in cards]

        with pytest.raises(ValueError):
            ContrastiveBatch(
                items=cards, identities=identities, positive_cliques=[[0, 0]]
            )
