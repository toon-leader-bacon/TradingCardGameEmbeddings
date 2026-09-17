import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.deck_box_dealer import DeckBoxDealer
from src.dojos.contrastive.dojo import ContrastiveDojo
from src.schema.card import GenericCard, GenericDeck, Provenance
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


def _deck() -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="deck",
        card_nocab_uuids=[uuid4()],
        provenance=None,
    )


class _ScriptedPairConstructor:
    """Test double: yields the given ContrastiveBatches in order,
    ignoring the actual decks/card_lookup it's called with."""

    def __init__(self, batches: list[ContrastiveBatch]) -> None:
        self._batches = iter(batches)

    def build(self, decks: list[GenericDeck], card_lookup: object) -> ContrastiveBatch:
        return next(self._batches)


class _RecordingLoss:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def calculate(self, item_embeddings, identities, positive_cliques) -> torch.Tensor:
        self.calls.append((item_embeddings, identities, positive_cliques))
        return torch.tensor(0.0)


def _dealer_with_decks(count: int) -> DeckBoxDealer:
    box = DeckBox()
    for _ in range(count):
        box.create(_deck())
    return DeckBoxDealer(box, GameId.MTG, split_ratios=[1, 0, 0], seed=1)


class TestTrainingData:
    def test_skips_a_degenerate_batch_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        good_card_a, good_card_b = _card("a"), _card("b")
        degenerate_card = _card("c")
        good_batch = ContrastiveBatch(
            items=[good_card_a, good_card_b],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )
        degenerate_batch = ContrastiveBatch(
            items=[degenerate_card], identities=[(uuid4(),)], positive_cliques=[[0]]
        )
        pair_constructor = _ScriptedPairConstructor(
            [good_batch, degenerate_batch, good_batch]
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(3),
            pair_constructor=pair_constructor,
            card_lookup=CardBinder(),
            decks_per_sample=1,
        )

        with caplog.at_level(logging.WARNING):
            batches = list(dojo.training_data())

        assert batches == [good_batch, good_batch]
        assert "degenerate" in caplog.text


class TestComputeLoss:
    def test_raises_on_length_mismatch(self) -> None:
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(1),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
        )
        batch = ContrastiveBatch(
            items=[_card(), _card()],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )

        with pytest.raises(ValueError):
            dojo.compute_loss([torch.randn(4)], batch)

    def test_delegates_to_the_injected_loss(self) -> None:
        recording_loss = _RecordingLoss()
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(1),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            contrastive_loss=recording_loss,
        )
        batch = ContrastiveBatch(
            items=[_card(), _card()],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )
        embeddings = [torch.randn(4), torch.randn(4)]

        dojo.compute_loss(embeddings, batch)

        called_embeddings, called_identities, called_positive_cliques = (
            recording_loss.calls[0]
        )
        assert called_embeddings == embeddings
        assert called_identities == batch.identities
        assert called_positive_cliques == batch.positive_cliques
