"""First concrete Dojo: classify which game a card came from.

See plans/training_pipeline.md and src/dojos/dojo.py for the Dojo
contract this implements. Simplest possible dojo, deliberately: labels
are just a card's own source_game field, so no join against any other
data source (a 17lands metric table, a deck, ...) is needed anywhere
in this dojo — the whole point of building this one first is to make
the CardSet/CardLookup/DojoBatch data flow between a dojo and
training/ concrete without any other complexity in the way.
"""

import random
from uuid import UUID

import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.dojo import CardCount, CardSet, DecoderHead, Dojo, DojoBatch, Split
from src.dojos.losses.loss import Loss
from src.encoder_model.embedding import Embedding
from src.schema.game_id import GameId

_GAME_ID_ORDER = tuple(GameId)


class GameClassificationDojo(Dojo[GameId]):
    """Predicts a single card's GameId from its own embedding.

    CardCount(1, 1) — the simplest arity a dojo can declare. Labels
    are GameId itself; no separate label type is introduced since
    GameId already is the label (src/schema/game_id.py). Internally,
    compute_loss adapts GameId labels to the integer class indices
    self._loss (a Loss[torch.Tensor, int]) expects — that adaptation
    is this dojo's own private detail, not part of the Dojo[GameId]
    contract it exposes outward.
    """

    def __init__(
        self,
        train_ratio: float,
        validate_ratio: float,
        loss: Loss[torch.Tensor, int],
        rng_seed: int,
    ) -> None:
        """Construct a dojo that splits examples by the given ratios.

        Inputs:
            train_ratio: fraction of eligible cards assigned to the
                train pool.
            validate_ratio: fraction of eligible cards assigned to the
                validate pool. The remainder goes to test.
            loss: the injected loss computation this dojo delegates to
                from compute_loss (e.g. CrossEntropyLoss).
            rng_seed: seed for this dojo's own shuffling/sampling, for
                reproducible splits and batches.
        Output: none (constructor).
        Side effects: none.
        Exceptions: raises ValueError if train_ratio + validate_ratio
            is not strictly less than 1, or either ratio is not in
            (0, 1).

        Example:
            >>> dojo = GameClassificationDojo(
            ...     train_ratio=0.8,
            ...     validate_ratio=0.1,
            ...     loss=CrossEntropyLoss(label_smoothing=0.0),
            ...     rng_seed=0,
            ... )
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if not (0.0 < validate_ratio < 1.0):
            raise ValueError(f"validate_ratio must be in (0, 1), got {validate_ratio}")
        if train_ratio + validate_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + validate_ratio must be < 1, "
                f"got {train_ratio} + {validate_ratio}"
            )
        self._train_ratio = train_ratio
        self._validate_ratio = validate_ratio
        self._loss = loss
        self._rng = random.Random(rng_seed)
        self._corpus: CardLookup | None = None
        self._decoder_head: DecoderHead | None = None
        self._train_uuids: list[UUID] = []
        self._pass_uuids: dict[Split, list[UUID]] = {}
        self._pass_index: dict[Split, int] = {}

    def card_count(self) -> CardCount:
        """Output: CardCount(1, 1) — this dojo is single-card-only."""
        return CardCount(minimum=1, maximum=1)

    def prepare_splits(self, corpus: CardLookup, held_out_cards: set[UUID]) -> None:
        """Split every card in corpus by this dojo's configured ratios.

        held_out_cards is honored: any uuid in held_out_cards is never
        placed in the train pool (it's shuffled into validate/test
        instead), since a game-classification example needs exactly
        one card and there's no larger CardSet for a held-out card to
        "contaminate" the way a deck-shaped example could. Caches
        corpus itself, so later next_batch calls can look up cards by
        uuid without needing corpus passed in again.

        Inputs:
            corpus: card lookup to enumerate and split, and to cache
                for next_batch's own later lookups.
            held_out_cards: nocab_uuids to keep out of the train pool.
        Output: none — populates this dojo's own internal train/test/
            validate uuid pools.
        Side effects: mutates this dojo's own internal state.
        Exceptions: none.
        """
        self._corpus = corpus
        all_uuids = list(corpus.all_uuids())
        self._rng.shuffle(all_uuids)

        eligible_for_train = [u for u in all_uuids if u not in held_out_cards]
        train_count = min(round(len(all_uuids) * self._train_ratio), len(eligible_for_train))
        train_uuids = eligible_for_train[:train_count]

        train_uuid_set = set(train_uuids)
        remaining = [u for u in all_uuids if u not in train_uuid_set]
        self._rng.shuffle(remaining)

        validate_count = round(len(all_uuids) * self._validate_ratio)
        validate_uuids = remaining[:validate_count]
        test_uuids = remaining[validate_count:]

        self._train_uuids = train_uuids
        self._pass_uuids = {Split.TEST: test_uuids, Split.VALIDATE: validate_uuids}
        self._pass_index = {Split.TEST: 0, Split.VALIDATE: 0}

    def split_size(self, split: Split) -> int:
        """Report how many cards are in one of this dojo's pools.

        Inputs:
            split: which pool to size.
        Output: that pool's card count, as of the last prepare_splits
            call.
        Side effects: none.
        Exceptions: raises RuntimeError if called before
            prepare_splits.
        """
        if self._corpus is None:
            raise RuntimeError("prepare_splits must be called before split_size")
        if split == Split.TRAIN:
            return len(self._train_uuids)
        return len(self._pass_uuids[split])

    def next_batch(self, split: Split, batch_size: int) -> DojoBatch[GameId]:
        """Draw batch_size cards from split, each its own length-1 CardSet.

        See Split's own docstring (src/dojos/dojo.py) for the TRAIN
        vs. TEST/VALIDATE sampling behavior this must honor.

        Inputs:
            split: which pool to draw from.
            batch_size: how many examples to draw.
        Output: a DojoBatch of length-1 CardSets, each labeled with
            that card's own source_game.
        Side effects: advances this dojo's internal cursor for split.
        Exceptions: raises RuntimeError if called before
            prepare_splits.

        Example:
            >>> dojo = GameClassificationDojo(0.8, 0.1, CrossEntropyLoss(0.0), rng_seed=0)
            >>> dojo.prepare_splits(corpus, held_out_cards=set())
            >>> batch = dojo.next_batch(Split.TRAIN, batch_size=4)
        """
        if self._corpus is None:
            raise RuntimeError("prepare_splits must be called before next_batch")
        if split == Split.TRAIN:
            uuids = self._sample_train_uuids(batch_size)
        else:
            uuids = self._next_pass_uuids(split, batch_size)
        return self._build_batch(uuids)

    def build_decoder_head(self, embedding_dim: int) -> DecoderHead:
        """Construct a fixed-size linear classifier over GameId.

        Caches the returned head on this dojo, so compute_loss can run
        it later.

        Inputs:
            embedding_dim: the encoder's output embedding width.
        Output: a fresh torch.nn.Linear(embedding_dim, len(GameId)).
        Side effects: caches the constructed head on this dojo.
        Exceptions: none.
        """
        self._decoder_head = torch.nn.Linear(embedding_dim, len(_GAME_ID_ORDER))
        return self._decoder_head

    def compute_loss(self, embeddings: list[list[Embedding]], labels: list[GameId]) -> torch.Tensor:
        """Run this dojo's decoder head, then its injected loss, over a batch.

        Inputs:
            embeddings: one length-1 list[Embedding] per example.
            labels: this batch's GameId labels, as returned by
                next_batch, same order.
        Output: a scalar loss tensor for this batch (self._loss's
            output, given this dojo's decoder head's predictions and
            the GameId labels converted to class indices).
        Side effects: none — does not call .backward() or step any
            optimizer.
        Exceptions: raises ValueError if any embeddings[i] does not
            hold exactly one Embedding, or len(embeddings) !=
            len(labels), or build_decoder_head hasn't been called yet.
        """
        if self._decoder_head is None:
            raise ValueError("build_decoder_head must be called before compute_loss")
        if len(embeddings) != len(labels):
            raise ValueError(
                f"embeddings and labels must be the same length, "
                f"got {len(embeddings)} and {len(labels)}"
            )
        for card_embeddings in embeddings:
            if len(card_embeddings) != 1:
                raise ValueError(
                    f"GameClassificationDojo is CardCount(1, 1); expected "
                    f"exactly one embedding per example, got {len(card_embeddings)}"
                )

        stacked = torch.stack([card_embeddings[0] for card_embeddings in embeddings])
        predictions = self._decoder_head(stacked)
        label_indices = [self._game_id_to_class_index(label) for label in labels]
        return self._loss.compute(predictions, label_indices)

    def _game_id_to_class_index(self, game_id: GameId) -> int:
        """Map a GameId to its fixed position in the decoder head's output.

        Private helper — single consumer is compute_loss (to build the
        target index this dojo's Loss expects) and, in reverse, would
        be used by any future inference code that reads
        build_decoder_head's logits back out.

        Inputs:
            game_id: the GameId to map.
        Output: that GameId's fixed class index, stable for the
            lifetime of this dojo instance.
        Side effects: none.
        Exceptions: none.
        """
        return _GAME_ID_ORDER.index(game_id)

    def _sample_train_uuids(self, batch_size: int) -> list[UUID]:
        """Sample batch_size uuids from the train pool, with replacement.

        Private helper — single consumer is next_batch() for
        Split.TRAIN.

        Inputs:
            batch_size: how many uuids to sample.
        Output: batch_size uuids drawn from this dojo's train pool.
        Side effects: consumes from this dojo's own rng state.
        Exceptions: raises IndexError if the train pool is empty.
        """
        return self._rng.choices(self._train_uuids, k=batch_size)

    def _next_pass_uuids(self, split: Split, batch_size: int) -> list[UUID]:
        """Advance split's cursor and return its next (possibly short) chunk.

        Private helper — single consumer is next_batch() for
        Split.TEST/Split.VALIDATE. Wraps to a freshly-shuffled order
        once a full pass over split completes.

        Inputs:
            split: TEST or VALIDATE.
            batch_size: how many uuids to return, except possibly
                fewer on the final chunk of a pass.
        Output: the next chunk of uuids for split, length batch_size
            except a pass's final (possibly short) chunk.
        Side effects: advances this dojo's internal cursor for split.
        Exceptions: none.
        """
        pool = self._pass_uuids[split]
        if not pool:
            return []

        index = self._pass_index[split]
        if index == 0:
            self._rng.shuffle(pool)

        end = index + batch_size
        chunk = pool[index:end]
        next_index = index + len(chunk)
        self._pass_index[split] = 0 if next_index >= len(pool) else next_index
        return chunk

    def _build_batch(self, uuids: list[UUID]) -> DojoBatch[GameId]:
        """Turn a list of uuids into a DojoBatch via this dojo's cached corpus.

        Private helper — single consumer is next_batch(), shared by
        all three Split cases once each has produced its uuid list.

        Inputs:
            uuids: uuids to look up via this dojo's cached CardLookup.
        Output: a DojoBatch of length-1 CardSets, labeled by each
            card's own source_game.
        Side effects: none.
        Exceptions: raises ValueError if any uuid has no card in this
            dojo's cached CardLookup.
        """
        assert self._corpus is not None  # guaranteed by next_batch's own guard
        card_sets = []
        labels = []
        for nocab_uuid in uuids:
            card = self._corpus.get_by_uuid(nocab_uuid)
            if card is None:
                raise ValueError(f"no card found for nocab_uuid {nocab_uuid}")
            card_sets.append(CardSet(cards=[card]))
            labels.append(card.source_game)
        return DojoBatch(card_sets=card_sets, labels=labels)
