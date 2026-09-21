"""Strategy: a batch's item embeddings -> one scalar contrastive loss.

See plans/contrastive_dojo.md's "ContrastiveLoss (swappable)" section.
A new Protocol, sibling to but distinct from NocabLoss
(src/dojos/loss/nocab_loss.py) - it cannot satisfy NocabLoss.calculate
(decoder_output, labels)'s row-independent shape, since InfoNCE/SupCon
need every item's embedding in the batch jointly. Out of scope for this
slice: a SupCon-style multi-positive variant.

The module-level `_pairwise_cosine_similarity`/`_valid_negative_mask`/
`_anchor_loss` functions below are the shared single-positive InfoNCE
math both SingleCardInfoNCELoss and MultiCardInfoNCELoss reduce to -
identical logic (PRINCIPLES.md section 2), not two classes that happen
to look alike. What differs between the two is only how each derives,
from its own item-level input shape, the flat embedding/identity pool
and the anchor -> positives/exclusions mapping these functions expect;
that derivation stays private to each class.
"""

from typing import Hashable, Protocol, Sequence, cast
from uuid import UUID

import torch

from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedMultiCardEmbedding,
    BatchedSingleCardEmbedding,
    Embedding,
    InputShape,
    output_shape_of,
)


def _pairwise_cosine_similarity(embeddings: list[Embedding]) -> torch.Tensor:
    """All-pairs cosine similarity over a flat pool of embeddings.

    Shared by every concrete ContrastiveLoss in this file - the pool is
    "one embedding per item" for SingleCardInfoNCELoss, or "one
    embedding per flattened card" for MultiCardInfoNCELoss; this
    function doesn't need to know which.

    Inputs:
        embeddings: a flat list of Embedding.
    Output: an (N, N) tensor, N = len(embeddings); entry [i, j] is the
        cosine similarity between embeddings[i] and embeddings[j].
    Side effects: none.
    Exceptions: none expected for non-empty, equal-dimension embeddings.
    """
    stacked = torch.stack(embeddings)
    normalized = torch.nn.functional.normalize(stacked, dim=1)
    return normalized @ normalized.T


def _valid_negative_mask(identities: Sequence[Hashable]) -> torch.Tensor:
    """Which (i, j) pairs may ever serve as anchor/negative.

    Shared by every concrete ContrastiveLoss in this file. Excludes
    i == j and any pair sharing an identical identity (the structural
    false-negative case - see this module's docstring). `identities`
    is whatever hashable key identifies "the same real-world thing" for
    one comparison unit - a whole ContrastiveBatch.identities tuple for
    a single-card item, or one bare card uuid for a flattened card.

    Inputs:
        identities: one hashable identity per comparison unit, same
            order as the embeddings _pairwise_cosine_similarity() was
            given.
    Output: an (N, N) boolean tensor, N = len(identities); True where
        unit j is a legitimate negative candidate for anchor i (a
        positive pair, or an anchor's own excluded group, may still
        separately override this at use-site - this mask governs
        exact-identity duplicate exclusion only).
    Side effects: none.
    Exceptions: none.
    """
    unit_count = len(identities)
    mask = torch.ones((unit_count, unit_count), dtype=torch.bool)
    for i in range(unit_count):
        mask[i, i] = False
        for j in range(i + 1, unit_count):
            if identities[i] == identities[j]:
                mask[i, j] = False
                mask[j, i] = False
    return mask


def _anchor_loss(
    anchor_index: int,
    anchor_positives: list[int],
    excluded_indices: set[int],
    similarity: torch.Tensor,
    valid_negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """One anchor's own single-positive InfoNCE loss, averaged over its
    positives if it has more than one.

    Shared by every concrete ContrastiveLoss in this file, called once
    per anchor unit that has at least one positive.

    Inputs:
        anchor_index: index of the anchor unit.
        anchor_positives: indices of anchor_index's positives (at least
            one).
        excluded_indices: indices that are neither a positive nor a
            valid negative for this anchor, beyond what
            valid_negative_mask already excludes - e.g. a multi-card
            anchor's own-item sibling cards. Empty for a ContrastiveLoss
            with no such notion (e.g. SingleCardInfoNCELoss).
        similarity: the full (N, N) pairwise cosine similarity matrix
            from _pairwise_cosine_similarity().
        valid_negative_mask: the full (N, N) mask from
            _valid_negative_mask().
        temperature: InfoNCE's softmax temperature.
    Output: a scalar loss tensor for this anchor: for each of its
        positives, -log(softmax(similarity[anchor] / temperature)
        restricted to {that positive} union anchor's valid negatives),
        averaged over its positives.
    Side effects: none.
    Exceptions: none expected (anchor_positives is non-empty by
        construction - see each class's calculate()).
    """
    excluded_from_negatives = set(anchor_positives) | excluded_indices
    unit_count = similarity.shape[0]
    valid_negative_indices = [
        j
        for j in range(unit_count)
        if bool(valid_negative_mask[anchor_index, j])
        and j not in excluded_from_negatives
    ]

    per_positive_losses: list[torch.Tensor] = []
    for positive_index in anchor_positives:
        # log_softmax over [this positive, every valid negative] is the
        # numerically stable form of -log(exp(pos)/(exp(pos) +
        # sum(exp(negatives)))) - the positive is always logit 0 in this
        # concatenation, so its log-probability is what we want.
        candidate_indices = [positive_index] + valid_negative_indices
        candidate_similarities = (
            similarity[anchor_index, candidate_indices] / temperature
        )
        log_probabilities = torch.log_softmax(candidate_similarities, dim=0)
        per_positive_losses.append(-log_probabilities[0])

    return torch.stack(per_positive_losses).mean()


class ContrastiveLoss(Protocol):
    """Computes one scalar loss from every item's embedding in a batch,
    jointly - the batch-level counterpart to NocabLoss's row-independent
    shape. Owned privately by a ContrastiveDojo."""

    def calculate(
        self,
        item_embeddings: BatchedModelOutput,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> torch.Tensor:
        """Compute this batch's contrastive loss.

        Inputs:
            item_embeddings: one item's embedding per entry, same
                order/length as identities - shape (single-card vs.
                multi-card item) is implementation-defined; a concrete
                ContrastiveLoss validates its own expected shape.
            identities: parallel to item_embeddings - see
                ContrastiveBatch.identities.
            positive_cliques: index groups into item_embeddings/
                identities - see ContrastiveBatch.positive_cliques.
        Output: a scalar loss tensor.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined (expected: ValueError on a
            length mismatch between item_embeddings and identities, or
            on an item_embeddings shape this implementation doesn't
            support).
        """
        ...


class SingleCardInfoNCELoss:
    """Concrete ContrastiveLoss for the single-card slice: single-positive
    InfoNCE over cosine similarity, computed once across the whole flat
    item pool (so every A-B/A-C/B-C deck-pair comparison falls out of
    one pairwise similarity matrix, never special-cased per deck pair).
    Exact-identity duplicates (see ContrastiveBatch.identities) are
    excluded from a given anchor's negative pool at this step, per the
    plan's deferred-to-loss-time policy - never precomputed or stored
    on the batch itself."""

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Inputs:
            temperature: InfoNCE's softmax temperature - lower sharpens
                the similarity distribution. 0.07 is CLIP's commonly
                cited default.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature

    def calculate(
        self,
        item_embeddings: BatchedModelOutput,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> torch.Tensor:
        """Mean anchor-wise single-positive InfoNCE loss over the batch.

        Inputs: see ContrastiveLoss.calculate(). item_embeddings must be
            single-card-shaped (one bare Embedding per item).
        Output: a scalar loss tensor - the mean, over every item that
            has at least one positive, of that item's own InfoNCE loss
            (its true positive scored against every valid negative -
            every other item, excluding itself and any exact-identity
            duplicate).
        Side effects: none.
        Exceptions: ValueError if len(item_embeddings) != len(identities),
            if item_embeddings isn't single-card-shaped, or if no item
            in the batch has a positive pair (loss is undefined).

        Example:
            >>> loss_fn = SingleCardInfoNCELoss()
            >>> loss_fn.calculate(item_embeddings, identities, positive_cliques)
        """
        if (
            item_embeddings
            and output_shape_of(item_embeddings[0]) != InputShape.SINGLE_CARD
        ):
            raise ValueError(
                "SingleCardInfoNCELoss expects single-card item embeddings "
                f"(got shape {output_shape_of(item_embeddings[0])} for the first item)"
            )
        if len(item_embeddings) != len(identities):
            raise ValueError(
                f"item_embeddings ({len(item_embeddings)}) and identities "
                f"({len(identities)}) must be the same length"
            )
        # Runtime shape already validated above - narrow the wide
        # ContrastiveLoss.calculate() parameter type back to the
        # concrete shape this class actually handles.
        single_card_embeddings = cast(BatchedSingleCardEmbedding, item_embeddings)

        similarity = _pairwise_cosine_similarity(single_card_embeddings)
        valid_negative_mask = _valid_negative_mask(identities)
        positives_by_anchor = self._positives_by_anchor(positive_cliques)

        # Accumulate each anchor's own InfoNCE loss, then average - an
        # anchor with no positive in this batch contributes nothing. No
        # item has an "excluded" group of its own in the single-card
        # shape (unlike MultiCardInfoNCELoss's own-item siblings), so
        # that argument is always empty here.
        per_anchor_losses: list[torch.Tensor] = []
        for anchor_index, anchor_positives in positives_by_anchor.items():
            per_anchor_losses.append(
                _anchor_loss(
                    anchor_index,
                    anchor_positives,
                    set(),
                    similarity,
                    valid_negative_mask,
                    self._temperature,
                )
            )

        if not per_anchor_losses:
            raise ValueError("No item in this batch has a positive pair")
        return torch.stack(per_anchor_losses).mean()

    def _positives_by_anchor(
        self, positive_cliques: list[list[int]]
    ) -> dict[int, list[int]]:
        """Expand positive_cliques into a per-anchor positives mapping.

        Private helper - single caller is calculate(). Every index in a
        clique is a positive for every other index in that same clique -
        each clique is a mutually-positive group, not a directed
        relation, so this is a pure expansion, not a derivation from
        any pairwise data.

        Inputs:
            positive_cliques: see calculate().
        Output: a mapping from anchor index to the list of item indices
            that are positives for that anchor. An index whose clique
            has no other member (size-1 clique) is simply absent as a
            key.
        Side effects: none.
        Exceptions: none.
        """
        result: dict[int, list[int]] = {}
        for group in positive_cliques:
            for index in group:
                other_indices = [other for other in group if other != index]
                if other_indices:
                    result[index] = other_indices
        return result


class MultiCardInfoNCELoss:
    """Concrete ContrastiveLoss for the multi-card slice: each item is a
    group of cards (see MultiCardPairConstructor), but the comparison
    unit is still the individual card, never a pooled whole-item vector.
    A card's positives are every card in a *different* item of its own
    item's positive clique (same source deck); cards in the anchor's own
    item are excluded entirely (neither positive nor negative), per the
    plan's explicit "an anchor is not a valid pair with other cards in
    its own set grouping" requirement - this is the one behavior
    genuinely new relative to SingleCardInfoNCELoss, expressed via
    _anchor_loss's excluded_indices argument."""

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Inputs:
            temperature: InfoNCE's softmax temperature - see
                SingleCardInfoNCELoss.__init__.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature

    def calculate(
        self,
        item_embeddings: BatchedModelOutput,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> torch.Tensor:
        """Mean per-card single-positive InfoNCE loss over the batch.

        Inputs: see ContrastiveLoss.calculate(). item_embeddings must be
            multi-card-shaped (one MultiCardEmbedding - a list of
            per-card Embedding - per item).
        Output: a scalar loss tensor - the mean, over every card that
            has at least one cross-item positive, of that card's own
            InfoNCE loss.
        Side effects: none.
        Exceptions: ValueError if len(item_embeddings) != len(identities),
            if item_embeddings isn't multi-card-shaped, or if no card in
            the batch has a positive pair (loss is undefined).

        Example:
            >>> loss_fn = MultiCardInfoNCELoss()
            >>> loss_fn.calculate(item_embeddings, identities, positive_cliques)
        """
        if (
            item_embeddings
            and output_shape_of(item_embeddings[0]) != InputShape.MULTI_CARD
        ):
            raise ValueError(
                "MultiCardInfoNCELoss expects multi-card item embeddings "
                f"(got shape {output_shape_of(item_embeddings[0])} for the first item)"
            )
        if len(item_embeddings) != len(identities):
            raise ValueError(
                f"item_embeddings ({len(item_embeddings)}) and identities "
                f"({len(identities)}) must be the same length"
            )
        # Runtime shape already validated above - narrow the wide
        # ContrastiveLoss.calculate() parameter type back to the
        # concrete shape this class actually handles.
        multi_card_embeddings = cast(BatchedMultiCardEmbedding, item_embeddings)

        flat_embeddings, flat_identities, item_of_card = self._flatten_items(
            multi_card_embeddings, identities
        )
        similarity = _pairwise_cosine_similarity(flat_embeddings)
        valid_negative_mask = _valid_negative_mask(flat_identities)
        positives_by_anchor, excluded_by_anchor = (
            self._positives_and_exclusions_by_anchor(item_of_card, positive_cliques)
        )

        # Accumulate each anchor card's own InfoNCE loss, then average -
        # a card with no cross-item positive in this batch contributes
        # nothing.
        per_card_losses: list[torch.Tensor] = []
        for anchor_index, anchor_positives in positives_by_anchor.items():
            per_card_losses.append(
                _anchor_loss(
                    anchor_index,
                    anchor_positives,
                    excluded_by_anchor[anchor_index],
                    similarity,
                    valid_negative_mask,
                    self._temperature,
                )
            )

        if not per_card_losses:
            raise ValueError("No card in this batch has a positive pair")
        return torch.stack(per_card_losses).mean()

    def _flatten_items(
        self,
        item_embeddings: BatchedMultiCardEmbedding,
        identities: list[tuple[UUID, ...]],
    ) -> tuple[list[Embedding], list[UUID], list[int]]:
        """Flatten every item's per-card embeddings/uuids into one flat
        card-level pool, tracking which item index each card came from.

        Private helper - single caller is calculate().

        Inputs:
            item_embeddings: see calculate() - one MultiCardEmbedding
                per item.
            identities: see calculate() - identities[i] is positionally
                parallel to item_embeddings[i] (ContrastiveBatch's own
                contract), so identities[i][k] is the uuid of the card
                at item_embeddings[i][k].
        Output: (flat_embeddings, flat_identities, item_of_card), all
            the same length (total card count across every item):
            flat_embeddings[k] and flat_identities[k] are one card's
            embedding/uuid, and item_of_card[k] is the item index (into
            item_embeddings/identities) that card came from.
        Side effects: none.
        Exceptions: none expected (identities[i] is assumed the same
            length as item_embeddings[i] - a ContrastiveBatch/
            MultiCardPairConstructor invariant, not re-validated here).
        """
        raise NotImplementedError

    def _positives_and_exclusions_by_anchor(
        self, item_of_card: list[int], positive_cliques: list[list[int]]
    ) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
        """Expand item-level positive_cliques down to card-level
        positive/excluded index sets for every flattened card.

        Private helper - single caller is calculate(). For a card at
        flat index k belonging to item i: its positives are every card
        belonging to a *different* item j in i's positive clique; its
        exclusions are every other card belonging to item i itself (see
        this class's own docstring).

        Inputs:
            item_of_card: item_of_card[k] is the item index flat card k
                came from - see _flatten_items().
            positive_cliques: see calculate().
        Output: (positives_by_anchor, excluded_by_anchor), both keyed by
            flat card index. A card with no cross-item positive in its
            clique is simply absent from positives_by_anchor (and
            _anchor_loss is never called for it, so excluded_by_anchor
            need not have an entry for it either).
        Side effects: none.
        Exceptions: none.
        """
        raise NotImplementedError
