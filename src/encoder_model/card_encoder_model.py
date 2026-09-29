"""CardEncoderModel: the shape handling shared by every card encoder.

Template Method. A subclass supplies two building blocks:

- embed_together(cards): cards that belong to one example (a deck, a
  group). Each card's embedding may depend on the others.
- embed_apart(cards): unrelated cards embedded in one batched pass. Each
  card's embedding depends only on that card.

The base class maps every input shape in src/schema/type_hints.py onto
those two methods. It offers one explicit method per shape, plus the
canonical forward(), which always treats its input as a batch.
"""

from abc import ABC, abstractmethod
from typing import List, Mapping, cast

import torch
import torch.nn as nn

from src.schema.card import GenericCard
from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedMultiCardEmbedding,
    BatchedMultiCardInput,
    BatchedMultiGroupEmbedding,
    BatchedMultiGroupInput,
    BatchedSingleCardEmbedding,
    BatchedSingleCardInput,
    BatchedTrainingInput,
    InputShape,
    MultiCardEmbedding,
    MultiCardInput,
    MultiGroupEmbedding,
    MultiGroupInput,
    SingleCardEmbedding,
    SingleCardInput,
    batched_input_shape_of,
)


class CardEncoderModel(nn.Module, ABC):
    """A card encoder. Every input shape maps onto embed_together and embed_apart.

    forward(x) is what Trainer and dojos call, via TrainableEncoder. It
    always reads x as a batch (BatchedTrainingInput): a flat list of
    cards is a batch of single cards, never one multi-card input.
    Callers that hold one unbatched example use the explicit forward_*
    method for its shape.
    """

    @abstractmethod
    def embed_together(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One embedding per card, contextualized within `cards` as one group.

        Inputs: cards (List[GenericCard]), one example's cards, non-empty
            (the base class never passes an empty group).
        Output: List[Tensor], same length and order as cards.
        Side effects: none beyond autograd.
        Exceptions: whatever the model's text encoder or head raises.
        """

    @abstractmethod
    def embed_apart(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One embedding per card; each depends only on its own card.

        Inputs: cards (List[GenericCard]), unrelated cards, non-empty
            (the base class never passes an empty list).
        Output: List[Tensor], same length and order as cards. No card's
            embedding depends on the other cards, in train or eval mode.
            In eval mode (no dropout), embedding i also equals
            embed_apart([cards[i]])[0], up to float rounding.
        Side effects: none beyond autograd.
        Exceptions: whatever the model's text encoder or head raises.
        """

    def encoder_only_state_dict(self) -> Mapping[str, torch.Tensor]:
        """Every weight of this model. Dojo decoder heads live in their
        dojos, not here. This is the artifact to publish.

        Inputs: none. Output: Mapping[str, Tensor], same as state_dict().
        Side effects: none. Exceptions: none.
        """
        return self.state_dict()

    def forward(self, x: BatchedTrainingInput) -> BatchedModelOutput:
        """Embed a batch, returning embeddings nested like x.

        Inputs: x (BatchedTrainingInput): a non-empty batch of single
            cards, multi-card inputs, or multi-group inputs.
        Output: BatchedModelOutput, one embedding per card, nested like x.
        Side effects: none beyond autograd.
        Exceptions: TypeError if x is a bare GenericCard (not a batch).
            ValueError if x is empty, malformed, or nested deeper than a
            BatchedMultiGroupInput. Also ValueError if the first example
            or its first group is empty: the shape is read from the first
            entry at each level, like Batch does. An empty group anywhere
            else becomes []. For other layouts, call the explicit method.

        Example:
            >>> embeddings = model([card_a, card_b])   # batch of 2 single cards
            >>> len(embeddings)
            2
        """
        # Classify one example; the batch's depth alone is ambiguous
        shape = batched_input_shape_of(x)

        # Dispatch to the batched method for that example shape
        if shape is InputShape.SINGLE_CARD:
            return self.forward_batched_single_card(cast(BatchedSingleCardInput, x))
        if shape is InputShape.MULTI_CARD:
            return self.forward_batched_multi_card(cast(BatchedMultiCardInput, x))
        if shape is InputShape.MULTI_GROUP:
            return self.forward_batched_multi_group(cast(BatchedMultiGroupInput, x))
        raise ValueError("Input is nested deeper than a BatchedMultiGroupInput")

    # region Explicit forward methods, one per input shape
    #
    # Contract shared by all six:
    # Output: embeddings nested exactly like x, one Tensor per card, in
    #   input order. An empty group or example becomes []. Subclasses are
    #   never called with zero cards; empty groups are common in real data
    #   (first-pick pools, rows with no blockers).
    # Side effects: none beyond autograd.
    # Exceptions: whatever the subclass's text encoder, head or attention
    #   raises. There are no shape checks: each method trusts x to have
    #   its own shape.

    def forward_single_card(self, x: SingleCardInput) -> SingleCardEmbedding:
        """One card -> its embedding (depth 0).

        Inputs: x (GenericCard). Output: Tensor.

        Example:
            >>> model.forward_single_card(card).shape
            torch.Size([256])
        """
        return self.embed_apart([x])[0]

    def forward_multi_card(self, x: MultiCardInput) -> MultiCardEmbedding:
        """One multi-card example -> one embedding per card, contextualized
        across the whole list (depth 1, unbatched).

        Inputs: x (List[GenericCard]), may be empty. Output: List[Tensor].

        Example:
            >>> deck = model.forward_multi_card([card_a, card_b])   # 2 tensors
        """
        return self._embed_group(x)

    def forward_multi_group(self, x: MultiGroupInput) -> MultiGroupEmbedding:
        """One multi-group example -> each group contextualized on its own
        (depth 2, unbatched).

        Inputs: x (List[List[GenericCard]]); groups may be empty.
        Output: List[List[Tensor]].

        Example:
            >>> model.forward_multi_group([[card_a, card_b], []])   # [[t, t], []]
        """
        return [self._embed_group(group) for group in x]

    def forward_batched_single_card(
        self, x: BatchedSingleCardInput
    ) -> BatchedSingleCardEmbedding:
        """A batch of unrelated single cards -> one embedding each, and no
        card sees another (depth 1, batched).

        Inputs: x (List[GenericCard]), may be empty. Output: List[Tensor].

        Example:
            >>> model.forward_batched_single_card([card_a, card_b])   # 2 tensors
        """
        return self.embed_apart(x) if x else []

    def forward_batched_multi_card(
        self, x: BatchedMultiCardInput
    ) -> BatchedMultiCardEmbedding:
        """A batch of multi-card examples -> each example contextualized on
        its own (depth 2, batched). This is the same computation as
        forward_multi_group, because groups never see each other either.

        Inputs: x (List[List[GenericCard]]); examples may be empty.
        Output: List[List[Tensor]].

        Example:
            >>> model.forward_batched_multi_card([[card_a, card_b], [card_c]])
        """
        return [self._embed_group(example) for example in x]

    def forward_batched_multi_group(
        self, x: BatchedMultiGroupInput
    ) -> BatchedMultiGroupEmbedding:
        """A batch of multi-group examples -> forward_multi_group on each
        (depth 3, batched).

        Inputs: x (List[List[List[GenericCard]]]).
        Output: List[List[List[Tensor]]].

        Example:
            >>> model.forward_batched_multi_group([[[card_a], []], [[card_b]]])
        """
        return [self.forward_multi_group(example) for example in x]

    # endregion

    def _embed_group(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """embed_together, except an empty group becomes [], so subclasses
        never see one.

        Inputs: cards (List[GenericCard]), possibly empty.
        Output: List[Tensor], same length and order.
        Side effects: none beyond autograd.
        Exceptions: as embed_together.
        """
        return self.embed_together(cards) if cards else []
