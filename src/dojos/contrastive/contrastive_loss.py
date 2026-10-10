"""Strategy: a batch's item embeddings -> one scalar contrastive loss.

See src/dojos/README.md's contrastive section. A new Protocol, sibling
to but distinct from NocabLoss
(src/dojos/loss/nocab_loss.py) - it cannot satisfy NocabLoss.calculate
(decoder_output, labels)'s row-independent shape, since InfoNCE/SupCon
need every item's embedding in the batch jointly. Out of scope: a
SupCon-style multi-positive variant.

This module holds the Protocol, DegenerateBatchError, and the shared
single-positive InfoNCE math (`check_item_embeddings`,
`pairwise_cosine_similarity`, `identity_negative_mask`, `anchor_loss`,
`mean_anchor_loss`, `mean_item_embeddings`, `mean_constant_logit_loss`)
that every InfoNCE style in styles/ reduces to - identical logic
(PRINCIPLES.md section 2), not classes that happen to look alike. What
differs between those styles is only how each derives, from its own
item-level input shape, the flat embedding/identity pool and the anchor
-> positives/exclusions mapping these functions expect; that derivation
stays private to each loss. Odd one out's loss is not InfoNCE (its
candidates are positions within one item) and uses only
`check_item_embeddings` and DegenerateBatchError.
"""

import math
from typing import (
    Hashable,
    Mapping,
    Protocol,
    Sequence,
)
from uuid import UUID

import torch

from src.schema.type_hints import (
    BatchedMultiCardEmbedding,
    BatchedModelOutput,
    Embedding,
    InputShape,
    output_shape_of,
)


def pairwise_cosine_similarity(embeddings: list[Embedding]) -> torch.Tensor:
    """All-pairs cosine similarity over a flat pool of embeddings.

    Shared by every InfoNCE style's loss (styles/) - the pool is
    "one embedding per item" for SingleCardInfoNCELoss, or "one
    embedding per flattened card" for CrossSliceCardInfoNCELoss; this
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


def mean_item_embeddings(item_embeddings: BatchedMultiCardEmbedding) -> list[Embedding]:
    """One vector per multi-card item: the mean of its card embeddings
    (for a one-card item, that card's embedding itself).

    Shared by every concrete ContrastiveLoss that compares whole items
    (the missing-card context, the slice-match slice).

    Inputs: item_embeddings, every item non-empty.
    Output: list[Embedding], one per item, same order.
    Side effects: none beyond autograd.
    Exceptions: none expected.

    Example:
        >>> mean_item_embeddings([[torch.ones(2), torch.zeros(2)]])
        [tensor([0.5000, 0.5000])]
    """
    return [torch.stack(item).mean(dim=0) for item in item_embeddings]


def identity_negative_mask(
    identities: Sequence[Hashable], device: torch.device
) -> torch.Tensor:
    """Which (i, j) pairs may ever serve as anchor/negative.

    Shared by every InfoNCE style's loss (styles/). Excludes
    i == j and any pair sharing an identical identity (the structural
    false-negative case - see this module's docstring). `identities`
    is whatever hashable key identifies "the same real-world thing" for
    one comparison unit - a whole ContrastiveBatch.identities tuple for
    a single-card item or a missing-card style item, or one bare card
    uuid for a flattened card.

    Inputs:
        identities: one hashable identity per comparison unit, same
            order as the embeddings pairwise_cosine_similarity() was
            given.
        device: where the similarity matrix lives; the mask is moved there.
    Output: an (N, N) boolean tensor on device, N = len(identities); True where
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
    # Filled on the CPU (one element write each), then moved once
    return mask.to(device)


def anchor_loss(
    anchor_index: int,
    anchor_positives: list[int],
    excluded_indices: set[int],
    similarity: torch.Tensor,
    valid_negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """One anchor's own single-positive InfoNCE loss, averaged over its
    positives if it has more than one.

    Shared by every InfoNCE style's loss (styles/), called once
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
            from pairwise_cosine_similarity().
        valid_negative_mask: the full (N, N) mask from
            identity_negative_mask().
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


class DegenerateBatchError(ValueError):
    """A well-formed batch whose loss is undefined: no unit has a positive,
    or no anchor has a negative (e.g. one surviving deck). Distinct from a
    malformed batch (a ValueError from a loss's own parsing), which is a
    pair-constructor bug; ContrastiveDojo skips only this one."""


def check_item_embeddings(
    item_embeddings: BatchedModelOutput,
    identities: Sequence[tuple[UUID, ...]],
    expected_shape: InputShape,
    loss_name: str,
) -> None:
    """The input checks every concrete ContrastiveLoss.calculate() starts
    with: each item has expected_shape, and one identity per item.

    Shared by every style's loss (styles/), odd one out included.

    Inputs: item_embeddings, identities (see ContrastiveLoss.calculate()),
        expected_shape (the item shape this loss reads), loss_name (for
        the error message).
    Output: none.
    Side effects: none.
    Exceptions: ValueError on a shape or length mismatch.
    """
    if item_embeddings and output_shape_of(item_embeddings[0]) != expected_shape:
        # e.g. InputShape.MULTI_CARD -> "multi-card"
        shape_text = expected_shape.name.lower().replace("_", "-")
        raise ValueError(
            f"{loss_name} expects {shape_text} item embeddings "
            f"(got shape {output_shape_of(item_embeddings[0])} for the first item)"
        )
    if len(item_embeddings) != len(identities):
        raise ValueError(
            f"item_embeddings ({len(item_embeddings)}) and identities "
            f"({len(identities)}) must be the same length"
        )


def mean_anchor_loss(
    positives_by_anchor: Mapping[int, list[int]],
    excluded_by_anchor: Mapping[int, set[int]],
    similarity: torch.Tensor,
    valid_negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """anchor_loss for every anchor, averaged: the reduction every
    concrete ContrastiveLoss.calculate() ends with.

    Shared by every InfoNCE style's loss (styles/).

    Inputs: positives_by_anchor (non-empty lists), excluded_by_anchor (a
        missing key means no extra exclusions), similarity and
        valid_negative_mask (see anchor_loss), temperature.
    Output: a scalar loss tensor.
    Side effects: none.
    Exceptions: DegenerateBatchError (a ValueError) if positives_by_anchor
        is empty (no anchor, loss undefined).
    """
    if not positives_by_anchor:
        raise DegenerateBatchError("no unit in this batch has a positive pair")
    per_anchor_losses = [
        anchor_loss(
            anchor_index,
            anchor_positives,
            excluded_by_anchor.get(anchor_index, set()),
            similarity,
            valid_negative_mask,
            temperature,
        )
        for anchor_index, anchor_positives in positives_by_anchor.items()
    ]
    return torch.stack(per_anchor_losses).mean()


def mean_constant_logit_loss(
    identities: Sequence[Hashable],
    positives_by_anchor: Mapping[int, list[int]],
    excluded_by_anchor: Mapping[int, set[int]],
) -> float:
    """The batch loss anchor_loss would give if every similarity were
    equal: the InfoNCE baseline for this batch shape (encoder-free).

    Shared by every InfoNCE style's loss (styles/), exactly as
    anchor_loss is. With constant logits, each of an anchor's per-positive
    terms is -log(1 / (1 + |N(a)|)) whatever the temperature, so

        baseline = mean over anchors a with P(a) non-empty of ln(1 + |N(a)|)
        N(a) = units - {a} - P(a) - E(a) - D(a)

    where P(a) = a's positives, E(a) = its extra exclusions (a multi-card
    anchor's own-item siblings; for a missing-card anchor, the other
    same-role units and decks missing the same card), D(a) = exact-identity duplicates of a
    (identity_negative_mask). Closed forms for duplicate-free batches with
    clique sizes k_c (items), N items in all:
        single-card: sum_c k_c ln(N - k_c + 1) / sum_c k_c
        multi-card, m cards per item: sum_c k_c ln(m (N - k_c) + 1) / sum_c k_c
    (sums over cliques with k_c >= 2). Counting from the batch directly
    (no closed form) keeps duplicates and short decks exact.

    Inputs:
        identities: one identity per comparison unit, as for
            identity_negative_mask.
        positives_by_anchor: anchor unit -> its positives (non-empty
            lists only), as each class's calculate() derives it.
        excluded_by_anchor: anchor unit -> extra exclusions; a missing
            key means none.
    Output: float > 0 when some anchor has a negative.
    Side effects: none.
    Exceptions: ValueError if positives_by_anchor is empty (no anchor,
        loss undefined) or every anchor has zero negatives (baseline 0,
        normalized loss undefined).
    """
    if not positives_by_anchor:
        raise DegenerateBatchError("no unit in this batch has a positive pair")
    unit_count = len(identities)
    per_anchor_losses: list[float] = []
    for anchor, positives in positives_by_anchor.items():
        not_negative = set(positives) | excluded_by_anchor.get(anchor, set())
        negative_count = sum(
            1
            for j in range(unit_count)
            if j != anchor
            and j not in not_negative
            and identities[j] != identities[anchor]
        )
        per_anchor_losses.append(math.log(1 + negative_count))
    result = math.fsum(per_anchor_losses) / len(per_anchor_losses)
    if result <= 0.0:
        raise DegenerateBatchError(
            "no anchor in this batch has a negative; baseline is 0"
        )
    return result


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

    def constant_logit_loss(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> float:
        """The loss calculate() would return for this batch if every
        similarity were equal (an encoder that learned nothing): the
        batch's baseline for normalized loss. Needs no embeddings.

        Inputs: identities, positive_cliques - see calculate().
        Output: float > 0.
        Side effects: none.
        Exceptions: ValueError if no unit has a positive, or no anchor
            has a negative (see mean_constant_logit_loss).
        """
        ...
