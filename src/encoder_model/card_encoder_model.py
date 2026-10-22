"""CardEncoderModel: the shape handling shared by every card encoder.

Template Method. forward() owns the one batched path every input shape
takes: flatten the batch's cards, one card_pipeline.embed call, scatter
into a padded (sequences, max_cards, d) tensor, then the subclass's
contextualize(cards, mask), then zero the padding. A subclass supplies
only contextualize. See src/schema/padded_embeddings.py for the output
layout.
"""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence, cast

import torch
import torch.nn as nn

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.feature_cache.pooled_feature_cache import (
    FeatureCacheStats,
    FeatureReservation,
)
from src.encoder_model.precision import Precision, inference_context
from src.schema.card import GenericCard
from src.schema.padded_embeddings import (
    ModelEmbeddings,
    PaddedEmbeddings,
    mask_from_lengths,
)
from src.schema.type_hints import (
    BatchedMultiCardEmbedding,
    BatchedMultiCardInput,
    BatchedMultiGroupEmbedding,
    BatchedMultiGroupInput,
    BatchedSingleCardInput,
    BatchedTrainingInput,
    InputShape,
    batched_input_shape_of,
)


class CardEncoderModel(nn.Module, ABC):
    """A card encoder. Every input shape maps onto one contextualize call.

    forward(x) is what Trainer and dojos call, via TrainableEncoder. It
    always reads x as a batch (BatchedTrainingInput): a flat list of
    cards is a batch of single cards, never one multi-card input. A
    caller holding one unbatched example wraps it in a one-example batch.

    Every model owns one CardEmbeddingPipeline (card_pipeline), set here so
    the cache entry points (warm, feature_cache_stats, close_feature_cache)
    live in one place.
    """

    def __init__(self, card_pipeline: CardEmbeddingPipeline) -> None:
        """Inputs: card_pipeline (registered as the submodule
            "card_pipeline", so state_dict keys keep that prefix).
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        super().__init__()
        self.card_pipeline: CardEmbeddingPipeline = card_pipeline

    def warm(
        self,
        cards: Sequence[GenericCard],
        precision: Precision,
        durable: bool | None = True,
    ) -> None:
        """Fill the card pipeline's feature cache with cards ahead of
        training; nothing happens without a cache.

        Inputs: cards (any number), precision (the run's autocast),
            durable (see CardEmbeddingPipeline.warm).
        Output: none.
        Side effects: see CardEmbeddingPipeline.warm.
        Exceptions: see CardEmbeddingPipeline.warm.

        Example:
            >>> model.warm(list(binder.all_cards(GameId.GWENT)), "fp16")
        """
        self.card_pipeline.warm(cards, precision, durable)

    def reserve(
        self,
        cards: Sequence[GenericCard],
        precision: Precision,
        durable: bool | None = None,
    ) -> FeatureReservation:
        """Pin and warm one batch's cards in the card pipeline's cache.

        Inputs: cards, precision, durable (see
            CardEmbeddingPipeline.reserve).
        Output: FeatureReservation (empty without a cache).
        Side effects: see CardEmbeddingPipeline.reserve.
        Exceptions: see CardEmbeddingPipeline.reserve.

        Example:
            >>> reservation = model.reserve(batch_cards, "fp16")
        """
        return self.card_pipeline.reserve(cards, precision, durable)

    def release(self, reservation: FeatureReservation) -> None:
        """Remove a reservation's pins (see CardEmbeddingPipeline.release).

        Inputs: reservation. Output: none.
        Side effects: see CardEmbeddingPipeline.release.
        Exceptions: see CardEmbeddingPipeline.release.

        Example:
            >>> model.release(reservation)
        """
        self.card_pipeline.release(reservation)

    def feature_cache_stats(self) -> FeatureCacheStats | None:
        """The card pipeline's cache totals, or None without a cache.

        Inputs: none. Output: FeatureCacheStats or None.
        Side effects: none. Exceptions: none.

        Example:
            >>> model.feature_cache_stats().avoided_fraction
            0.97
        """
        cache = self.card_pipeline.feature_cache
        return None if cache is None else cache.stats()

    def close_feature_cache(self) -> None:
        """Close the card pipeline's feature cache (its database
        connection); nothing without a cache.

        Inputs: none. Output: none. Side effects: closes the cache, which
            must not be used after. Exceptions: none.

        Example:
            >>> model.close_feature_cache()
        """
        cache = self.card_pipeline.feature_cache
        if cache is not None:
            cache.close()

    @abstractmethod
    def contextualize(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Contextualize each sequence's cards within that sequence.

        The one step a subclass supplies (Template Method); forward owns
        every other step.
        Inputs: cards, (sequences, max_cards, d) base embeddings,
            left-aligned, padding slots zero; mask, (sequences, max_cards)
            bool, True = real card. A sequence may be empty (all False).
        Output: (sequences, max_cards, d). A real card's row may depend
            only on the real cards of its own sequence. Padding slots may
            hold anything; forward zeroes them.
        Side effects: none beyond autograd.
        Exceptions: whatever the subclass's layers raise.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """The width of one card embedding (every Tensor this model returns).

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """

    def isolated_embeddings(
        self, cards: Sequence[GenericCard], precision: Precision = "fp32"
    ) -> torch.Tensor:
        """Every card embedded on its own, for storage and analysis.

        Context-free by construction: a batch of single cards, so each
        card is its own length-one sequence and row i depends only on
        cards[i] (for MultiCardModel this is its no-context encoding).
        Deterministic: eval mode, so no dropout.

        Inputs: cards (Sequence[GenericCard]), from any onboarded game, may
            be empty; precision (Precision) for the forward pass.
        Output: float32 Tensor (len(cards), embedding_dim) on the CPU,
            detached; (0, embedding_dim) for no cards.
        Side effects: none lasting; the model's train/eval mode is switched
            to eval for the call and restored, even on error.
        Exceptions: whatever the text encoder, head or attention raise;
            entering autocast can raise for a dtype the device does not
            support.

        Example:
            >>> model.isolated_embeddings([card_a, card_b], "fp16").shape
            torch.Size([2, 256])
        """
        # No cards: no forward pass, just an empty table of the right width
        if not cards:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        # Embed each card apart, without training, at `precision`
        with inference_context(self, precision):
            padded = self._padded_sequences(
                [[card] for card in cards], InputShape.SINGLE_CARD
            )
        # One float32 CPU table (autocast may have produced fp16)
        return padded.cards[:, 0].float().cpu()

    def encoder_only_state_dict(self) -> Mapping[str, torch.Tensor]:
        """Every weight of this model. Dojo decoder heads live in their
        dojos, not here. This is the artifact to publish.

        Inputs: none. Output: Mapping[str, Tensor], same as state_dict().
        Side effects: none. Exceptions: none.
        """
        return self.state_dict()

    def forward(self, x: BatchedTrainingInput) -> ModelEmbeddings:
        """Embed a batch: one card_pipeline.embed call and one
        contextualize call, whatever the shape.

        Inputs: x (BatchedTrainingInput): a non-empty batch of single
            cards, multi-card inputs, or multi-group inputs.
        Output: ModelEmbeddings.
            - single cards -> PaddedEmbeddings, SINGLE_CARD, max_cards 1:
              each card is its own sequence, so no card sees another;
            - multi-card -> PaddedEmbeddings, MULTI_CARD: one sequence per
              example;
            - multi-group -> BatchedMultiGroupEmbedding: one sequence per
              group across all examples, regrouped; an empty group is [].
        Side effects: none beyond autograd.
        Exceptions: TypeError if x is a bare GenericCard (not a batch).
            ValueError if x is empty, nested deeper than a
            BatchedMultiGroupInput, or its first example (or that
            example's first group) is empty: the shape is read from the
            first entry at each level, like Batch does.

        Example:
            >>> model([card_a, card_b]).cards.shape   # batch of 2 single cards
            torch.Size([2, 1, 256])
        """
        # Classify one example; the batch's depth alone is ambiguous
        shape = batched_input_shape_of(x)

        # Single cards and multi-card examples: rows are the sequences
        if shape is InputShape.SINGLE_CARD:
            single_cards = cast(BatchedSingleCardInput, x)
            return self._padded_sequences([[card] for card in single_cards], shape)
        if shape is InputShape.MULTI_CARD:
            return self._padded_sequences(cast(BatchedMultiCardInput, x), shape)

        # Multi-group: every group is a sequence, then regroup per example
        if shape is InputShape.MULTI_GROUP:
            examples = cast(BatchedMultiGroupInput, x)
            groups = [group for example in examples for group in example]
            padded = self._padded_sequences(groups, InputShape.MULTI_CARD)
            return _regrouped(padded, [len(example) for example in examples])
        raise ValueError("Input is nested deeper than a BatchedMultiGroupInput")

    def _padded_sequences(
        self, sequences: Sequence[Sequence[GenericCard]], shape: InputShape
    ) -> PaddedEmbeddings:
        """The batched path every shape takes: embed every card at once,
        scatter into the padded layout, contextualize, zero the padding.

        Private helper - callers are forward and isolated_embeddings.
        Inputs: sequences (each may be empty, but at least one card in
            all), shape (SINGLE_CARD for one-card sequences, MULTI_CARD
            otherwise).
        Output: PaddedEmbeddings over sequences, rows in input order.
        Side effects: none beyond autograd.
        Exceptions: as card_pipeline.embed and contextualize.
        """
        # Flatten every card in input order, keeping lengths on the host
        lengths = tuple(len(sequence) for sequence in sequences)
        flat_cards = [card for sequence in sequences for card in sequence]

        # One embed call for the whole batch, then scatter row-major
        base = self.card_pipeline.embed(flat_cards)  # (num_cards, d)
        mask = mask_from_lengths(lengths, base.device)
        padded = base.new_zeros((len(lengths), mask.shape[1], base.shape[1]))
        padded[mask] = base

        # One contextualize call; then restore the zero-padding invariant
        contextualized = self.contextualize(padded, mask)
        cards = contextualized.masked_fill(~mask.unsqueeze(-1), 0.0)
        return PaddedEmbeddings(cards, mask, shape, lengths)


def _regrouped(
    groups: PaddedEmbeddings, group_counts: Sequence[int]
) -> BatchedMultiGroupEmbedding:
    """Split a padded batch of groups back into multi-group examples.

    Private helper - single caller is CardEncoderModel.forward.
    Inputs: groups (one row per group, all examples' groups in order),
        group_counts (groups per example; sums to len(groups)).
    Output: BatchedMultiGroupEmbedding, via groups.to_nested(); an empty
        group becomes [].
    Side effects: none beyond autograd. Exceptions: none.
    """
    result: BatchedMultiGroupEmbedding = []
    nested = cast(BatchedMultiCardEmbedding, groups.to_nested())
    # Hand each example its next group_count groups, in order
    start = 0
    for group_count in group_counts:
        result.append(nested[start : start + group_count])
        start += group_count
    return result
