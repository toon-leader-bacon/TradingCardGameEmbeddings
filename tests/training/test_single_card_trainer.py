from datetime import datetime, timezone
from uuid import uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import CardCount, Split
from src.dojos.game_classification.game_classification_dojo import GameClassificationDojo
from src.dojos.losses.cross_entropy_loss import CrossEntropyLoss
from src.encoder_model.single_card.toy_single_card_model import ToySingleCardModel
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.training.single_card_trainer import SingleCardTrainer


def _card(name: str, source_game: GameId) -> GenericCard:
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


def _binder() -> CardBinder:
    binder = CardBinder()
    for i in range(8):
        binder.add(_card(f"mtg-{i}", GameId.MTG))
    for i in range(8):
        binder.add(_card(f"pokemon-{i}", GameId.POKEMON))
    return binder


def _trainer(binder: CardBinder, held_out_cards: set = frozenset()) -> SingleCardTrainer:
    dojo = GameClassificationDojo(
        train_ratio=0.5,
        validate_ratio=0.2,
        loss=CrossEntropyLoss(label_smoothing=0.0),
        rng_seed=0,
    )
    model = ToySingleCardModel(embedding_dim=8, vocab_size=256)
    return SingleCardTrainer(
        model=model,
        dojo=dojo,
        corpus=binder,
        held_out_cards=set(held_out_cards),
        learning_rate=1e-2,
    )


class _FixedArityDojo:
    def card_count(self) -> CardCount:
        return CardCount(minimum=2, maximum=2)


class TestInit:
    def test_rejects_a_multi_card_dojo(self) -> None:
        with pytest.raises(ValueError):
            SingleCardTrainer(
                model=ToySingleCardModel(embedding_dim=8, vocab_size=256),
                dojo=_FixedArityDojo(),  # type: ignore[arg-type]
                corpus=_binder(),
                held_out_cards=set(),
                learning_rate=1e-2,
            )

    def test_prepares_the_dojos_splits(self) -> None:
        binder = _binder()
        dojo = GameClassificationDojo(
            train_ratio=0.5, validate_ratio=0.2, loss=CrossEntropyLoss(0.0), rng_seed=0
        )
        SingleCardTrainer(
            model=ToySingleCardModel(embedding_dim=8, vocab_size=256),
            dojo=dojo,
            corpus=binder,
            held_out_cards=set(),
            learning_rate=1e-2,
        )

        assert dojo.split_size(Split.TRAIN) > 0


class TestTrain:
    def test_returns_finite_test_and_validate_losses(self) -> None:
        trainer = _trainer(_binder())

        result = trainer.train(num_steps=3, batch_size=4)

        assert torch.isfinite(result.test_loss)
        assert torch.isfinite(result.validate_loss)
        assert result.test_loss.dim() == 0
        assert result.validate_loss.dim() == 0

    def test_training_steps_change_model_weights(self) -> None:
        binder = _binder()
        dojo = GameClassificationDojo(
            train_ratio=0.5, validate_ratio=0.2, loss=CrossEntropyLoss(0.0), rng_seed=0
        )
        model = ToySingleCardModel(embedding_dim=8, vocab_size=256)
        trainer = SingleCardTrainer(
            model=model,
            dojo=dojo,
            corpus=binder,
            held_out_cards=set(),
            learning_rate=1e-1,
        )
        before = model._table.weight.clone()

        trainer.train(num_steps=5, batch_size=4)

        assert not torch.equal(before, model._table.weight)

    def test_zero_train_steps_leaves_model_weights_unchanged(self) -> None:
        binder = _binder()
        dojo = GameClassificationDojo(
            train_ratio=0.5, validate_ratio=0.2, loss=CrossEntropyLoss(0.0), rng_seed=0
        )
        model = ToySingleCardModel(embedding_dim=8, vocab_size=256)
        trainer = SingleCardTrainer(
            model=model,
            dojo=dojo,
            corpus=binder,
            held_out_cards=set(),
            learning_rate=1e-1,
        )
        before = model._table.weight.clone()

        trainer.train(num_steps=0, batch_size=4)

        assert torch.equal(before, model._table.weight)
