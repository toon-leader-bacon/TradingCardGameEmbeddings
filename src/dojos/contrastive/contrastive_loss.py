"""Strategy: a batch's item embeddings -> one scalar contrastive loss.

See src/dojos/README.md's contrastive section. A new Protocol, sibling
to but distinct from NocabLoss
(src/dojos/loss/nocab_loss.py) - it cannot satisfy NocabLoss.calculate
(decoder_output, labels)'s row-independent shape, since InfoNCE/SupCon
need every item's embedding in the batch jointly. Out of scope: a
SupCon-style multi-positive variant.

This module holds the Protocol, DegenerateBatchError, and the shared
single-positive InfoNCE math every InfoNCE style in styles/ reduces to
(identical logic, PRINCIPLES.md section 2):

- `pairwise_cosine_similarity`, `mean_item_embeddings` and
  `check_item_embeddings` turn a batch into an (N, N) similarity over
  its comparison units;
- `same_key_mask`, `unit_keys`, `clique_keys` and `AnchorMasks` state
  who is each anchor's positive and negative, as (N, N) boolean masks;
- `masked_info_nce_loss` and `constant_logit_baseline` reduce the masks
  (and the similarity) to the loss and its baseline, all anchors at
  once, in a few tensor kernels whatever N is.

What differs between those styles is only how each derives, from its
own item-level input shape, the flat unit pool and its AnchorMasks;
that derivation stays private to each loss (its _anchor_masks). Odd one
out's loss is not InfoNCE (its candidates are positions within one
item) and uses only `check_item_embeddings` and DegenerateBatchError.
"""

import math
from dataclasses import dataclass
from typing import (
    Hashable,
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


def same_key_mask(keys: torch.Tensor) -> torch.Tensor:
    """Which pairs of units share a key: one broadcast comparison.

    Shared by every InfoNCE style's _anchor_masks (styles/), which turn
    each O(N) key vector (clique, item, role, identity) into an (N, N)
    relation with it.

    Inputs: keys, an (N,) int64 tensor on any device.
    Output: an (N, N) bool tensor on keys' device; [i, j] is
        keys[i] == keys[j], so the diagonal is True.
    Side effects: none.
    Exceptions: ValueError if keys is not 1-D int64.

    Example:
        >>> same_key_mask(torch.tensor([0, 0, 1]))
        tensor([[ True,  True, False],
                [ True,  True, False],
                [False, False,  True]])
    """
    if keys.dim() != 1 or keys.dtype != torch.int64:
        raise ValueError(
            f"keys must be 1-D int64, got shape {tuple(keys.shape)} {keys.dtype}"
        )
    return keys.unsqueeze(1) == keys.unsqueeze(0)


def unit_keys(units: Sequence[Hashable], device: torch.device) -> torch.Tensor:
    """One dense int64 key per unit, equal for equal units (keys are
    numbered in first-seen order).

    Shared by every InfoNCE style's _anchor_masks: identities become
    identity keys, a missing card's identity its missing-card key.

    Inputs: units, any hashable values; device, where the keys go.
    Output: an (N,) int64 tensor on device, N = len(units).
    Side effects: none.
    Exceptions: none.

    Example:
        >>> unit_keys(["a", "b", "a"], torch.device("cpu"))
        tensor([0, 1, 0])
    """
    key_of_unit: dict[Hashable, int] = {}
    result = [key_of_unit.setdefault(unit, len(key_of_unit)) for unit in units]
    return torch.tensor(result, dtype=torch.int64, device=device)


def clique_keys(
    positive_cliques: Sequence[Sequence[int]], unit_count: int, device: torch.device
) -> torch.Tensor:
    """Each unit's clique as a key; a unit in no clique gets a key of its
    own, so it has no positive.

    Shared by every InfoNCE style's _anchor_masks. Parses the cliques:
    they must be disjoint, since a unit's positives are exactly the
    other members of its one clique.

    Inputs:
        positive_cliques: index groups into the units (see
            ContrastiveBatch.positive_cliques).
        unit_count: N, the number of units the indices point into.
        device: where the keys go.
    Output: an (N,) int64 tensor on device. Clique c's members get key
        c; a unit in no clique gets a key >= len(positive_cliques) that no
        other unit has.
    Side effects: none.
    Exceptions: ValueError if an index is outside 0..N-1 or a unit is in
        two cliques.

    Example:
        >>> clique_keys([[0, 2], [1]], 4, torch.device("cpu"))
        tensor([0, 1, 0, 5])
    """
    # Every unit starts with a key of its own, above every clique index
    result = list(range(len(positive_cliques), len(positive_cliques) + unit_count))
    seen: set[int] = set()

    # Each clique stamps its index on its members
    for clique_index, clique in enumerate(positive_cliques):
        for unit in clique:
            # Negative indices would wrap around silently
            if not 0 <= unit < unit_count:
                raise ValueError(f"clique index {unit} is outside 0..{unit_count - 1}")
            if unit in seen:
                raise ValueError(f"unit {unit} is in more than one clique")
            seen.add(unit)
            result[unit] = clique_index
    return torch.tensor(result, dtype=torch.int64, device=device)


@dataclass(frozen=True)
class AnchorMasks:
    """One batch's anchor relations over N comparison units: row =
    anchor, column = the unit it is compared with. What every InfoNCE
    style reduces its batch to, so calculate() and constant_logit_loss()
    read the same relation (AnchorMasks.of derives the negatives).

    positives: (N, N) bool; [a, p] is True when p is one of a's
        positives. An anchor is a row with any True (see anchors).
    negatives: (N, N) bool; [a, j] is True when j is one of a's
        negatives.

    Invariants, checked in __post_init__: both (N, N) bool on one
    device, both diagonals False, positives and negatives disjoint. The
    value checks read one value back to the host per batch (the old loop
    read N^2), so masks can't be built on the META device. The fields
    are frozen but the tensors are not copied: don't edit them in place.
    """

    positives: torch.Tensor
    negatives: torch.Tensor

    def __post_init__(self) -> None:
        """Check the invariants above.

        Inputs: none (the fields).
        Output: none.
        Side effects: one host sync (the value checks).
        Exceptions: ValueError if an invariant does not hold.
        """
        # Validate metadata first: shape, dtype, device (no sync)
        _check_mask_layout(self.positives, self.negatives)

        # Then the values: no self pair, no unit both positive and negative
        has_self_pair = (
            self.positives.diagonal().any() | self.negatives.diagonal().any()
        )
        has_overlap = (self.positives & self.negatives).any()
        if bool(has_self_pair | has_overlap):
            raise ValueError(
                "AnchorMasks needs False diagonals and disjoint positives and negatives"
            )

    @property
    def anchors(self) -> torch.Tensor:
        """(N,) bool: the units with at least one positive."""
        return self.positives.any(dim=1)

    @classmethod
    def of(
        cls,
        positives: torch.Tensor,
        ignored: torch.Tensor | None,
        identity_keys: torch.Tensor,
    ) -> "AnchorMasks":
        """Build the masks from what a style knows: who is positive and
        who is ignored. Every other unit is a negative, except the anchor
        itself and its exact-identity duplicates (structural false
        negatives):

            negatives = not (self or positives or ignored or same identity)

        Identity only ever removes negatives: a positive that shares its
        anchor's identity (e.g. two copies of one card drawn from the
        same deck) stays a positive.

        Inputs:
            positives: (N, N) bool; its diagonal is cleared here.
            ignored: (N, N) bool, units neither positive nor negative for
                the row's anchor (e.g. a multi-card anchor's own-item
                siblings); may overlap positives, and positives win.
                None means no unit is ignored.
            identity_keys: (N,) int64 on the same device, as unit_keys
                gives for the units' identities.
        Output: AnchorMasks on positives' device.
        Side effects: one host sync (__post_init__).
        Exceptions: ValueError on a shape, dtype or device mismatch.

        Example:
            >>> cpu = torch.device("cpu")
            >>> keys = clique_keys([[0, 1], [2, 3]], 4, cpu)
            >>> masks = AnchorMasks.of(same_key_mask(keys), None, unit_keys("abcd", cpu))
            >>> masks.negatives[0].tolist()
            [False, False, True, True]
        """
        # Validate inputs: one layout for all three relations
        same_identity = same_key_mask(identity_keys)
        _check_mask_layout(positives, same_identity)
        _check_mask_layout(positives, ignored)
        own = torch.eye(positives.shape[0], dtype=torch.bool, device=positives.device)
        positives = positives & ~own

        # Every unit not positive, ignored, self or a duplicate is a negative
        not_negative = own | positives | same_identity
        if ignored is not None:
            not_negative = not_negative | ignored
        return cls(positives=positives, negatives=~not_negative)


def _check_mask_layout(mask: torch.Tensor, other: torch.Tensor | None) -> None:
    """Raise ValueError unless mask is square bool and other (if any)
    matches it in shape, dtype and device.

    Private helper - callers are AnchorMasks.__post_init__ and
    AnchorMasks.of.
    Inputs: mask, other (None skips the comparison).
    Output: none.
    Side effects: none (metadata only, no host sync).
    Exceptions: ValueError on a mismatch.
    """
    if mask.dim() != 2 or mask.shape[0] != mask.shape[1] or mask.dtype != torch.bool:
        raise ValueError(
            f"a mask must be square bool, got shape {tuple(mask.shape)} {mask.dtype}"
        )
    if other is None:
        return
    if (other.shape, other.dtype, other.device) != (
        mask.shape,
        mask.dtype,
        mask.device,
    ):
        raise ValueError(
            f"masks disagree: {tuple(mask.shape)} {mask.dtype} on {mask.device} "
            f"vs {tuple(other.shape)} {other.dtype} on {other.device}"
        )


def masked_info_nce_loss(
    similarity: torch.Tensor, masks: AnchorMasks, temperature: float
) -> torch.Tensor:
    """Single-positive InfoNCE for every anchor and positive at once: the
    reduction every InfoNCE style's calculate() ends with. With logits
    L = similarity / temperature (float32):

        loss(a, p) = logaddexp(L[a, p], LSE_{j in Neg(a)} L[a, j]) - L[a, p]
                   = softplus(LSE_{j in Neg(a)} L[a, j] - L[a, p])
        loss       = mean over anchors of (mean over a's positives)

    i.e. -log softmax of the positive against itself and a's negatives.
    The softplus form is computed: it has no cancellation when the loss
    is tiny (the logaddexp form loses it to float32 rounding). An anchor
    with no negative scores 0.

    Inputs:
        similarity: (N, N) pairwise cosine similarity
            (pairwise_cosine_similarity), any float dtype.
        masks: AnchorMasks on similarity's device.
        temperature: the softmax temperature, > 0 (callers check it).
    Output: a scalar float32 tensor, differentiable in similarity.
    Side effects: one host sync (the degenerate check); the reduction
        itself reads nothing back.
    Exceptions: DegenerateBatchError if no unit has a positive;
        ValueError if similarity's shape or device differ from the masks'.

    Example:
        >>> # Two decks of two orthogonal cards: each anchor's positive
        >>> # ties its 2 negatives at similarity 0, so the loss is ln 3
        >>> keys = clique_keys([[0, 1], [2, 3]], 4, cpu)
        >>> masks = AnchorMasks.of(same_key_mask(keys), None, unit_keys("abcd", cpu))
        >>> masked_info_nce_loss(torch.eye(4), masks, 1.0)
        tensor(1.0986)
    """
    # Validate inputs
    _check_similarity_layout(similarity, masks)
    anchors = masks.anchors
    if not bool(anchors.any()):
        raise DegenerateBatchError("no unit in this batch has a positive pair")

    # Every anchor's negative log-sum-exp, then each positive against it
    logits = similarity.float() / temperature
    negative_lse = _masked_row_logsumexp(logits, masks.negatives)
    per_pair = torch.nn.functional.softplus(negative_lse.unsqueeze(1) - logits)

    # Mean over each anchor's positives, then over anchors (sums, not
    # boolean indexing, which would sync for its data-dependent shape)
    per_pair = torch.where(masks.positives, per_pair, 0.0)
    positive_counts = masks.positives.sum(dim=1).clamp(min=1)
    per_anchor = per_pair.sum(dim=1) / positive_counts
    return (per_anchor * anchors).sum() / anchors.sum()


def _check_similarity_layout(similarity: torch.Tensor, masks: AnchorMasks) -> None:
    """Raise ValueError unless similarity has the masks' shape and device.

    Private helper - single caller is masked_info_nce_loss.
    Inputs: similarity, masks. Output: none.
    Side effects: none. Exceptions: ValueError on a mismatch.
    """
    expected = (masks.positives.shape, masks.positives.device)
    if (similarity.shape, similarity.device) != expected:
        raise ValueError(
            f"similarity is {tuple(similarity.shape)} on {similarity.device}, "
            f"masks are {tuple(expected[0])} on {expected[1]}"
        )


def _masked_row_logsumexp(logits: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Per row, the log-sum-exp of logits where keep is True. Masked
    entries are the float32 minimum, not -inf, so a row with nothing kept
    gives a finite value near that minimum: the later softplus of it is
    0, and so is its gradient (not NaN).

    Private helper - single caller is masked_info_nce_loss.
    Inputs: logits, (N, N) float32; keep, (N, N) bool, same device.
    Output: (N,) float32.
    Side effects: none. Exceptions: none.
    """
    floor = torch.finfo(logits.dtype).min
    return torch.logsumexp(logits.masked_fill(~keep, floor), dim=1)


def constant_logit_baseline(masks: AnchorMasks) -> float:
    """The loss masked_info_nce_loss would give if every similarity were
    equal: the InfoNCE baseline for this batch shape (encoder-free).
    Each of an anchor's per-positive terms is then ln(1 + |Neg(a)|),
    whatever the temperature, so

        baseline = mean over anchors a of ln(1 + |Neg(a)|)

    Shared by every InfoNCE style's constant_logit_loss(). Closed forms
    for duplicate-free batches with clique sizes k_c, N items:
        single-card: sum_c k_c ln(N - k_c + 1) / sum_c k_c
        multi-card, m cards per item: sum_c k_c ln(m (N - k_c) + 1) / sum_c k_c
    Counting from the masks keeps duplicates and short decks exact.

    Inputs: masks (any device; CPU in practice).
    Output: float > 0. Negatives are counted per anchor as int64; the
        logs and the mean are taken in Python (math.log, math.fsum).
    Side effects: none.
    Exceptions: DegenerateBatchError if no unit has a positive, or no
        anchor has a negative (baseline 0, normalized loss undefined).

    Example:
        >>> constant_logit_baseline(masks)  # 3 decks x 2 cards: ln 5
        1.609...
    """
    # Count each anchor's negatives
    anchors = masks.anchors
    if not bool(anchors.any()):
        raise DegenerateBatchError("no unit in this batch has a positive pair")
    negative_counts: list[int] = masks.negatives[anchors].sum(dim=1).tolist()

    # Average ln(1 + count) in float64
    result = math.fsum(math.log(1 + count) for count in negative_counts) / len(
        negative_counts
    )
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
            has a negative (see constant_logit_baseline).
        """
        ...
