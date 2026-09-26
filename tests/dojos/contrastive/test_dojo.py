import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.dojo import ContrastiveDojo
from src.dojos.dojo import BatchBudget, Dojo
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
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

    cards_per_deck = 1

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
    # A throwaway index file - these tests don't care about a dealer's
    # persisted index location, only its dealt-decks behavior, so a
    # fresh tempdir per call avoids threading a tmp_path fixture
    # through every one of this module's test methods.
    index_path = Path(tempfile.mkdtemp()) / "dealer.db"
    return DeckBoxDealer(box, GameId.MTG, index_path, split_ratios=[1, 0, 0], seed=1)


_BUDGET = BatchBudget(max_cost=10, cost_of=lambda card: 1)


class TestBatches:
    def test_skips_a_degenerate_batch_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        good_card_a, good_card_b = _card("a"), _card("b")
        degenerate_card = _card("c")
        good_batch = ContrastiveBatch(
            inputs=[good_card_a, good_card_b],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )
        degenerate_batch = ContrastiveBatch(
            inputs=[degenerate_card], identities=[(uuid4(),)], positive_cliques=[[0]]
        )
        pair_constructor = _ScriptedPairConstructor(
            [good_batch, degenerate_batch, good_batch]
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=pair_constructor,
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
            decks_per_sample=2,
        )

        with caplog.at_level(logging.WARNING):
            batches = list(dojo.batches(Split.TRAIN, _BUDGET))

        assert batches == [good_batch, good_batch]
        assert "degenerate" in caplog.text


class TestComputeLoss:
    def test_raises_on_length_mismatch(self) -> None:
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(1),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
        )
        batch = ContrastiveBatch(
            inputs=[_card(), _card()],
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
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
            contrastive_loss=recording_loss,
        )
        batch = ContrastiveBatch(
            inputs=[_card(), _card()],
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


class TestBudget:
    def test_raises_when_the_budget_cannot_fit_two_decks(self) -> None:
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
        )

        with pytest.raises(ValueError):
            list(dojo.batches(Split.TRAIN, BatchBudget(1, lambda card: 1)))

    def test_raises_when_a_built_batch_exceeds_the_budget(self) -> None:
        cards = [_card(), _card(), _card()]
        heavy = ContrastiveBatch(
            inputs=cards,  # type: ignore[arg-type]
            identities=[(c.nocab_uuid,) for c in cards],
            positive_cliques=[[0, 1, 2]],
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([heavy]),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
            decks_per_sample=2,
        )

        with pytest.raises(ValueError):
            list(dojo.batches(Split.TRAIN, BatchBudget(2, lambda card: 1)))

    def test_max_examples_stops_after_that_many_decks(self) -> None:
        good = ContrastiveBatch(
            inputs=[_card(), _card()],
            identities=[(uuid4(),), (uuid4(),)],
            positive_cliques=[[0, 1]],
        )
        dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([good, good, good]),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
            decks_per_sample=2,
        )

        batches = list(dojo.batches(Split.TRAIN, _BUDGET, max_examples=4))

        assert len(batches) == 2

    def test_satisfies_the_dojo_protocol_and_counts_decks(self) -> None:
        dojo: Dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(6),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
        )

        assert dojo.example_count(Split.TRAIN) == 6
        assert list(dojo.trainable_parameters()) == []


class TestVersionCheck:
    def test_raises_when_dealer_has_no_recorded_card_binder_version(self) -> None:
        with pytest.raises(ValueError, match="CardBinder"):
            ContrastiveDojo(
                dealer=_dealer_with_decks(2),
                pair_constructor=_ScriptedPairConstructor([]),
                card_lookup=CardBinder(),
                holdout=HoldoutSpec.no_holdout(),
            )

    def test_passes_when_dealers_recorded_version_matches(self, tmp_path: Path) -> None:
        binder = CardBinder()
        box = DeckBox()
        box.create(_deck())
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, binder.version_for(GameId.MTG))
        dealer = DeckBoxDealer(
            DeckBox.load([path]),
            GameId.MTG,
            tmp_path / "dealer.db",
            split_ratios=[1, 0, 0],
        )

        dojo = ContrastiveDojo(
            dealer=dealer,
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=binder,
            holdout=HoldoutSpec.no_holdout(),
        )

        assert dojo.example_count(Split.TRAIN) == 1

    def test_raises_when_dealers_recorded_version_is_stale(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        box = DeckBox()
        box.create(_deck())
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, "stale-version")
        dealer = DeckBoxDealer(
            DeckBox.load([path]),
            GameId.MTG,
            tmp_path / "dealer.db",
            split_ratios=[1, 0, 0],
        )

        with pytest.raises(ValueError, match="CardBinder"):
            ContrastiveDojo(
                dealer=dealer,
                pair_constructor=_ScriptedPairConstructor([]),
                card_lookup=binder,
                holdout=HoldoutSpec.no_holdout(),
            )

    def test_strict_version_check_false_skips_a_real_mismatch(self) -> None:
        dojo: Dojo = ContrastiveDojo(
            dealer=_dealer_with_decks(2),
            pair_constructor=_ScriptedPairConstructor([]),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            strict_version_check=False,
        )

        assert dojo.example_count(Split.TRAIN) == 2
