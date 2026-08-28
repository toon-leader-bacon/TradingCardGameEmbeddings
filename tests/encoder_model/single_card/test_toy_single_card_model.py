from datetime import datetime, timezone
from uuid import uuid4

import torch

from src.encoder_model.single_card.toy_single_card_model import ToySingleCardModel
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id="src-1",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestEmbeddingDim:
    def test_reports_constructor_value(self) -> None:
        model = ToySingleCardModel(embedding_dim=16, vocab_size=64)

        assert model.embedding_dim == 16


class TestForward:
    def test_output_shape_matches_embedding_dim(self) -> None:
        model = ToySingleCardModel(embedding_dim=8, vocab_size=64)

        embedding = model(_card("Lightning Bolt"))

        assert embedding.shape == torch.Size([8])

    def test_same_name_always_maps_to_the_same_embedding(self) -> None:
        model = ToySingleCardModel(embedding_dim=8, vocab_size=64)

        first = model(_card("Lightning Bolt"))
        second = model(_card("Lightning Bolt"))

        assert torch.equal(first, second)

    def test_different_names_usually_map_to_different_embeddings(self) -> None:
        model = ToySingleCardModel(embedding_dim=8, vocab_size=4096)

        bolt = model(_card("Lightning Bolt"))
        bear = model(_card("Grizzly Bears"))

        assert not torch.equal(bolt, bear)

    def test_embedding_is_a_trainable_parameter(self) -> None:
        model = ToySingleCardModel(embedding_dim=8, vocab_size=64)

        embedding = model(_card("Lightning Bolt"))

        assert embedding.requires_grad
