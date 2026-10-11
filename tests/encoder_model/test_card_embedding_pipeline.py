"""CardEmbeddingPipeline: the stages chain by width, and embed runs them in
order (text encoder -> enrichment -> pooling -> head)."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import torch

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.embedding_head import LinearEmbeddingHead
from src.encoder_model.enrichment.token_enrichment import TokenEnrichment
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.text_encoder import TokenEncoding
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder, fake_card_pipeline


class _TokenDoubling(TokenEnrichment):
    """Test enrichment: each token concatenated with itself (width x2)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=2 * input_dim)

    def forward(self, encoding: TokenEncoding) -> TokenEncoding:
        doubled = torch.cat([encoding.hidden_states, encoding.hidden_states], dim=-1)
        return TokenEncoding(
            hidden_states=doubled, attention_mask=encoding.attention_mask
        )


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={"name": name, "text": f"rules for {name}"},
        provenance=Provenance(
            DataSource.SCRYFALL, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


CARDS = [_card("Alpha"), _card("Bravo"), _card("Charlie")]


def test_embed_is_head_of_pooling_of_the_encoded_text() -> None:
    torch.manual_seed(0)
    pipeline = fake_card_pipeline(5)
    encoding = FakeTextEncoder().encode([serialize_card_to_json(c) for c in CARDS])
    expected = pipeline.embedding_head(pipeline.pooling(encoding))
    assert torch.allclose(pipeline.embed(CARDS), expected)
    assert pipeline.output_dim == 5


def test_an_enrichment_runs_between_encoder_and_pooling() -> None:
    torch.manual_seed(0)
    pipeline = CardEmbeddingPipeline(
        FakeTextEncoder(),
        _TokenDoubling(HIDDEN),
        MeanPooling(2 * HIDDEN),
        LinearEmbeddingHead(2 * HIDDEN, 5),
    )
    encoding = FakeTextEncoder().encode([serialize_card_to_json(c) for c in CARDS])
    doubled = _TokenDoubling(HIDDEN)(encoding)
    expected = pipeline.embedding_head(pipeline.pooling(doubled))
    assert torch.allclose(pipeline.embed(CARDS), expected)


def test_each_card_is_embedded_independently() -> None:
    torch.manual_seed(0)
    pipeline = fake_card_pipeline(5)
    alone = pipeline.embed(CARDS[:1])
    in_batch = pipeline.embed(CARDS)[:1]
    assert torch.allclose(alone, in_batch, atol=1e-6)


@pytest.mark.parametrize(
    "stages, message",
    [
        (
            (None, MeanPooling(HIDDEN + 1), LinearEmbeddingHead(HIDDEN + 1, 5)),
            "text encoder",
        ),
        ((None, MeanPooling(HIDDEN), LinearEmbeddingHead(HIDDEN + 1, 5)), "pooling"),
        (
            (
                _TokenDoubling(HIDDEN),
                MeanPooling(HIDDEN),
                LinearEmbeddingHead(HIDDEN, 5),
            ),
            "enrichment",
        ),
        (
            (
                _TokenDoubling(HIDDEN + 1),
                MeanPooling(2 * HIDDEN),
                LinearEmbeddingHead(2 * HIDDEN, 5),
            ),
            "text encoder",
        ),
    ],
)
def test_widths_that_do_not_chain_are_refused(stages: tuple, message: str) -> None:
    enrichment, pooling, head = stages
    with pytest.raises(ValueError, match=message):
        CardEmbeddingPipeline(FakeTextEncoder(), enrichment, pooling, head)
