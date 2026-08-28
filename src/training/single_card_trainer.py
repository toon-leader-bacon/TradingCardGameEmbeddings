"""The first Trainer implementation: one SingleCardModel, one single-card Dojo.

See plans/training_pipeline.md. Deliberately scoped to exactly one
active dojo — multi-dojo TrainingRegime dispatch is an explicit open
question in that plan (no implementation exists yet to drive), not
something to build ahead of a second dojo actually needing it.
"""

import itertools
import math
from typing import Any
from uuid import UUID

import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.dojo import CardCount, CardSet, Dojo, Split
from src.encoder_model.embedding import Embedding
from src.encoder_model.single_card.single_card_model import SingleCardModel
from src.training.trainer import EvaluationResult, Trainer


class SingleCardTrainer(Trainer):
    """Drives one SingleCardModel through one CardCount(1, 1) dojo.

    model is expected to also be a torch.nn.Module (see
    SingleCardModel's own docstring), so its parameters can be
    registered with this trainer's optimizer alongside the dojo's own
    decoder head.
    """

    def __init__(
        self,
        model: SingleCardModel,
        dojo: Dojo[Any],
        corpus: CardLookup,
        held_out_cards: set[UUID],
        learning_rate: float,
    ) -> None:
        """Construct a trainer wiring model and dojo together.

        Calls dojo.prepare_splits(corpus, held_out_cards) and
        dojo.build_decoder_head(model.embedding_dim), then registers
        both model's and the returned decoder head's parameters with a
        fresh optimizer.

        Inputs:
            model: the SingleCardModel to train.
            dojo: the single-card dojo to train against — already
                constructed, with whatever Loss it needs already
                injected; this trainer never constructs a dojo itself.
            corpus: card lookup passed straight through to
                dojo.prepare_splits — a loaded CardBinder satisfies
                this directly, no wrapper needed (see
                card_binder/card_lookup.py). Not held onto or used
                again after construction.
            held_out_cards: passed through to dojo.prepare_splits
                unchanged.
            learning_rate: optimizer learning rate.
        Output: none (constructor).
        Side effects: calls dojo.prepare_splits, mutating dojo's own
            internal state.
        Exceptions: raises ValueError if dojo.card_count() is not
            CardCount(1, 1).

        Example:
            >>> trainer = SingleCardTrainer(
            ...     ToySingleCardModel(embedding_dim=32, vocab_size=4096),
            ...     GameClassificationDojo(0.8, 0.1, CrossEntropyLoss(0.0), rng_seed=0),
            ...     CardBinder.load([Path("data/final/cards/mtg.jsonl")]),
            ...     held_out_cards=set(),
            ...     learning_rate=1e-3,
            ... )
        """
        if dojo.card_count() != CardCount(minimum=1, maximum=1):
            raise ValueError("SingleCardTrainer only drives CardCount(1, 1) dojos")

        self._model = model
        self._dojo = dojo
        dojo.prepare_splits(corpus, held_out_cards)

        self._decoder_head = dojo.build_decoder_head(model.embedding_dim)
        self._optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), self._decoder_head.parameters()),
            lr=learning_rate,
        )

    def train(self, num_steps: int, batch_size: int) -> EvaluationResult:
        """Run num_steps train-split steps, then score test/validate once each.

        Inputs:
            num_steps: how many train-split optimizer steps to run.
            batch_size: how many examples per step/pass batch.
        Output: this run's EvaluationResult.
        Side effects: mutates self._model's and self._decoder_head's
            weights; advances self._optimizer's state.
        Exceptions: none beyond what its stub helpers raise.

        Example:
            >>> trainer = SingleCardTrainer(
            ...     ToySingleCardModel(embedding_dim=32, vocab_size=4096),
            ...     GameClassificationDojo(
            ...         train_ratio=0.8,
            ...         validate_ratio=0.1,
            ...         loss=CrossEntropyLoss(label_smoothing=0.1),
            ...         rng_seed=0,
            ...     ),
            ...     CardBinder.load([Path("data/final/cards/mtg.jsonl")]),
            ...     held_out_cards=set(),
            ...     learning_rate=1e-3,
            ... )
            >>> trainer.train(num_steps=1000, batch_size=64)
        """
        for _ in range(num_steps):
            self._run_train_step(batch_size)

        test_loss = self._score_full_split(Split.TEST, batch_size)
        validate_loss = self._score_full_split(Split.VALIDATE, batch_size)
        return EvaluationResult(test_loss=test_loss, validate_loss=validate_loss)

    def _embed_card_sets(self, card_sets: list[CardSet]) -> list[list[Embedding]]:
        """Run self._model over every card in every CardSet.

        Private helper — single consumer is _run_train_step()/
        _score_full_split(). Calls self._model(card) rather than
        self._model.forward(card) directly (standard torch.nn.Module
        convention — forward hooks fire this way).

        Inputs:
            card_sets: one CardSet per example.
        Output: one list[Embedding] per input CardSet, same order and
            length, embeddings[i][j] = self._model(card_sets[i].cards[j]).
        Side effects: none — embeddings remain attached to the
            autograd graph for the caller's own backward() call.
        Exceptions: none.
        """
        return [[self._model(card) for card in card_set.cards] for card_set in card_sets]

    def _run_train_step(self, batch_size: int) -> torch.Tensor:
        """Run one full TRAIN step: draw a batch, embed, loss, backward, step.

        Private helper — single consumer is train()'s train-phase
        loop. Composes self._dojo.next_batch(Split.TRAIN, batch_size),
        _embed_card_sets, and self._dojo.compute_loss.

        Inputs:
            batch_size: how many examples to draw for this step.
        Output: this step's scalar loss tensor (post-backward).
        Side effects: mutates self._model's and self._decoder_head's
            weights via self._optimizer.step(); zeroes gradients
            before returning.
        Exceptions: none beyond what its composed calls raise.
        """
        batch = self._dojo.next_batch(Split.TRAIN, batch_size)
        embeddings = self._embed_card_sets(batch.card_sets)
        loss = self._dojo.compute_loss(embeddings, batch.labels)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss

    def _score_full_split(self, split: Split, batch_size: int) -> torch.Tensor:
        """Run one full no-gradient pass over split, returning its mean loss.

        Private helper — single consumer is train()'s test/validate
        phases (split is Split.TEST or Split.VALIDATE, never
        Split.TRAIN). Draws exactly
        ceil(self._dojo.split_size(split) / batch_size) batches via
        self._dojo.next_batch, under torch.no_grad().

        Inputs:
            split: Split.TEST or Split.VALIDATE.
            batch_size: how many examples per batch within this pass.
        Output: the mean loss across every batch in one full pass over
            split. Zero if split is empty.
        Side effects: none — no gradient computed, no optimizer step.
        Exceptions: none beyond what its composed calls raise.
        """
        num_batches = math.ceil(self._dojo.split_size(split) / batch_size)
        losses = []
        with torch.no_grad():
            for _ in range(num_batches):
                batch = self._dojo.next_batch(split, batch_size)
                embeddings = self._embed_card_sets(batch.card_sets)
                losses.append(self._dojo.compute_loss(embeddings, batch.labels))

        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
