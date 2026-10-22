"""Strategy: a batch's item embeddings -> one scalar contrastive loss.

See src/dojos/README.md's contrastive section. A new Protocol, sibling
to but distinct from NocabLoss
(src/dojos/loss/nocab_loss.py) - it cannot satisfy NocabLoss.calculate
(decoder_output, labels)'s row-independent shape, since InfoNCE/SupCon
need every item's embedding in the batch jointly. Out of scope: a
SupCon-style multi-positive variant.

This module holds the Protocol, DegenerateBatchError, and the shared
single-positive InfoNCE math every InfoNCE style in styles/ reduces to:
one ComparisonMasks (which unit pairs are positives, which are valid
negatives), built on the CPU from identities and cliques, read by both
info_nce_loss (the loss, one batched computation) and constant_logit_baseline
(the baseline). What differs between styles is only which comparison
units they score (an item's first card, its mean, every flat card) and
how they derive the masks; that derivation stays private to each loss.
Odd one out's loss is not InfoNCE (its candidates are positions within
one item) and uses only check_item_embeddings and DegenerateBatchError.
"""

import math
from dataclasses import dataclass
from typing import Hashable, Protocol, Sequence
from uuid import UUID

import torch

from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape
from src.utils.device_copy import to_device


class DegenerateBatchError(ValueError):
    """A well-formed batch whose loss is undefined: no unit has a positive,
    or no anchor has a negative (e.g. one surviving deck). Distinct from a
    malformed batch (a ValueError from a loss's own parsing), which is a
    pair-constructor bug; ContrastiveDojo skips only this one."""


@dataclass(frozen=True, eq=False)  # tensors have no truth value to compare
class ComparisonMasks:
    """Which pairs of comparison units are positives and which are valid
    negatives, for one batch. Both on the CPU; info_nce_loss moves them
    to the device once.

    positive: (N, N) bool; (i, j) True when j is a positive of anchor i.
        Never on the diagonal.
    negative: (N, N) bool; (i, j) True when j is a valid negative of
        anchor i: not i itself, not a positive, not an exact-identity
        duplicate, and not one of the style's own exclusions.
    """

    positive: torch.Tensor
    negative: torch.Tensor

    def __post_init__(self) -> None:
        """Check the masks are square, the same size, bool, on the CPU,
        and disjoint, with an empty diagonal.

        Inputs: the fields. Output: none.
        Side effects: none (CPU tensors, so no device sync).
        Exceptions: ValueError on any violation.
        """
        # Validate: square, matching, bool, on the CPU
        size = tuple(self.positive.shape)
        if len(size) != 2 or size[0] != size[1] or tuple(self.negative.shape) != size:
            raise ValueError(
                f"masks must be one square size, got {size} and "
                f"{tuple(self.negative.shape)}"
            )
        for mask in (self.positive, self.negative):
            if mask.dtype != torch.bool or mask.device.type != "cpu":
                raise ValueError("masks must be bool tensors on the CPU")

        # Validate: disjoint, and no unit pairs with itself
        if bool((self.positive & self.negative).any()):
            raise ValueError("a pair cannot be both a positive and a negative")
        if bool(self.positive.diagonal().any() or self.negative.diagonal().any()):
            raise ValueError("a unit cannot be its own positive or negative")

    @property
    def unit_count(self) -> int:
        """N. Inputs: none. Output: int. Side effects: none.
        Exceptions: none."""
        return int(self.positive.shape[0])

    def require_some_anchor(self) -> None:
        """Raise unless some unit has a positive.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: DegenerateBatchError if no row of positive is set.
        """
        if not bool(self.positive.any()):
            raise DegenerateBatchError("no unit in this batch has a positive pair")


def same_key_mask(keys: Sequence[Hashable]) -> torch.Tensor:
    """Which pairs of units share a key: the tensor comparison every mask
    is built from (identities, items, roles, missing cards).

    Shared by every InfoNCE style. Keys are mapped to integer ids in one
    pass, then compared as ids[:, None] == ids[None, :]: no O(N^2) Python
    loop.
    Inputs: keys, one hashable per comparison unit.
    Output: (N, N) bool on the CPU, True where keys[i] == keys[j]
        (including the diagonal).
    Side effects: none. Exceptions: none.

    Example:
        >>> same_key_mask(["a", "b", "a"])
        tensor([[ True, False,  True], [False,  True, False], [ True, False,  True]])
    """
    # One integer id per distinct key, in first-seen order
    id_of_key: dict[Hashable, int] = {}
    ids = torch.tensor(
        [id_of_key.setdefault(key, len(id_of_key)) for key in keys], dtype=torch.long
    )
    return ids.unsqueeze(1) == ids.unsqueeze(0)


def clique_positive_mask(
    positive_cliques: Sequence[Sequence[int]], unit_count: int
) -> torch.Tensor:
    """Every other member of an index's clique is its positive.

    Shared by every InfoNCE style. Cliques are mutually-positive groups,
    so this is a pure expansion.
    Inputs: positive_cliques (disjoint index groups into 0 .. unit_count -
        1), unit_count.
    Output: (unit_count, unit_count) bool on the CPU; (i, j) True when i
        and j share a clique and i != j. An index in no clique, or in a
        size-1 clique, has an empty row.
    Side effects: none.
    Exceptions: ValueError if an index is out of range or in two cliques.

    Example:
        >>> clique_positive_mask([[0, 1], [2]], 3).sum().item()
        2
    """
    # Label each index with its clique; -1 means no clique
    clique_of = torch.full((unit_count,), -1, dtype=torch.long)
    for clique_index, clique in enumerate(positive_cliques):
        for index in clique:
            if not 0 <= index < unit_count:
                raise ValueError(
                    f"clique index {index} out of range 0..{unit_count - 1}"
                )
            if clique_of[index] != -1:
                raise ValueError(f"index {index} is in more than one clique")
            clique_of[index] = clique_index

    # Same clique, a real clique, and not the unit itself
    same_clique = clique_of.unsqueeze(1) == clique_of.unsqueeze(0)
    in_a_clique = (clique_of >= 0).unsqueeze(1)
    return same_clique & in_a_clique & ~torch.eye(unit_count, dtype=torch.bool)


def info_nce_masks(
    positive: torch.Tensor,
    identities: Sequence[Hashable],
    excluded: torch.Tensor | None,
) -> ComparisonMasks:
    """Assemble ComparisonMasks: the negatives are everything that is not
    the anchor itself, a positive, an identity duplicate or excluded.

    Shared by every InfoNCE style; each builds positive and excluded its
    own way, then calls this.
    Inputs: positive (N, N) bool CPU; identities, one per unit (exact
        duplicates are never negatives: the structural false-negative
        case); excluded (N, N) bool CPU, the style's own extra exclusions
        (own-item siblings, same-role units), or None for none.
    Output: ComparisonMasks.
    Side effects: none. Exceptions: as ComparisonMasks.__post_init__.

    Example:
        >>> info_nce_masks(clique_positive_mask([[0, 1]], 3), ids, None).negative
    """
    # Everything starts as a candidate negative
    unit_count = positive.shape[0]
    negative = ~torch.eye(unit_count, dtype=torch.bool)

    # Remove positives, identity duplicates and the style's exclusions
    negative &= ~positive
    negative &= ~same_key_mask(identities)
    if excluded is not None:
        negative &= ~excluded
    return ComparisonMasks(positive, negative)


def info_nce_loss(
    units: torch.Tensor, masks: ComparisonMasks, temperature: float
) -> torch.Tensor:
    """Mean single-positive InfoNCE over every anchor with a positive, as
    one batched computation.

    Shared by every InfoNCE style. For each (anchor i, positive p):
        loss = -(l[i, p] - logaddexp(l[i, p], logsumexp(l[i, negative[i]])))
    with l = cosine(units, units) / temperature. Averaged over each
    anchor's positives, then over the anchors that have one (the same
    weighting as averaging per-anchor losses). An anchor with no negative
    contributes 0 with a finite, zero gradient: no reduction ever runs
    over an all -inf row (its NaN gradient would poison the batch).
    Inputs: units (N, d) on the device; masks (N units); temperature > 0.
    Output: scalar tensor.
    Side effects: non-blocking host-to-device copies of the masks; no
        device sync.
    Exceptions: DegenerateBatchError if no anchor has a positive (read
        from the CPU masks); ValueError if units has a row count other
        than masks.unit_count.

    Example:
        >>> info_nce_loss(torch.randn(4, 8), masks, 0.07)
        tensor(1.2345, grad_fn=...)
    """
    # Validate inputs on the host
    masks.require_some_anchor()
    _require_unit_count(units, masks)
    positive = to_device(masks.positive, units.device)
    negative = to_device(masks.negative, units.device)

    # Scaled cosine logits, and each anchor's log-sum-exp over its negatives
    logits = _cosine_logits(units, temperature)
    negative_lse = _negative_log_sum_exp(logits, negative)  # (N,), finite

    # Per (anchor, positive) loss; anchors without negatives give exactly 0
    pair_losses = _pair_losses(logits, negative_lse, negative.any(dim=1))

    # Mean over each anchor's positives, then over anchors with positives
    return _mean_over_anchors(pair_losses, positive)


def constant_logit_baseline(masks: ComparisonMasks) -> float:
    """The loss info_nce_loss would give if every similarity were equal:
    the InfoNCE baseline for this batch shape (encoder-free).

    Shared by every InfoNCE style. With constant logits each per-positive
    term is ln(1 + |N(a)|) whatever the temperature, so the baseline is
    the mean over anchors with a positive of ln(1 + negatives of a).
    Closed forms for duplicate-free batches with clique sizes k_c, N
    items: single-card sum_c k_c ln(N - k_c + 1) / sum_c k_c; multi-card
    with m cards per item, sum_c k_c ln(m (N - k_c) + 1) / sum_c k_c.
    Counting from the masks keeps duplicates and short decks exact.
    Inputs: masks.
    Output: float > 0.
    Side effects: none (CPU masks).
    Exceptions: DegenerateBatchError if no anchor has a positive, or no
        anchor has a negative (baseline 0, normalized loss undefined).

    Example:
        >>> constant_logit_baseline(single_card_masks)  # 3 decks x 2 items
        1.609...
    """
    # Validate: some anchor exists
    masks.require_some_anchor()

    # ln(1 + negative count) per anchor that has a positive
    anchors = masks.positive.any(dim=1)
    negative_counts = masks.negative[anchors].sum(dim=1).tolist()
    result = math.fsum(math.log1p(count) for count in negative_counts) / len(
        negative_counts
    )

    # A zero baseline means no anchor had a negative
    if result <= 0.0:
        raise DegenerateBatchError(
            "no anchor in this batch has a negative; baseline is 0"
        )
    return result


def check_item_embeddings(
    item_embeddings: PaddedEmbeddings,
    identities: Sequence[tuple[UUID, ...]],
    expected_shape: InputShape,
    loss_name: str,
) -> None:
    """The input checks every concrete ContrastiveLoss.calculate() starts
    with: the expected shape and one identity per item.

    An identity's length is not checked against the item's card count
    here: a composite loss legitimately passes a pooled item's whole sorted
    identity as one single-card key. A loss that reads positions (card k
    is identities[i][k]: cross-slice, odd one out, card in contexts) also
    calls require_card_aligned_identities.

    Shared by every style's loss (styles/), odd one out included.
    Inputs: item_embeddings, identities (see ContrastiveLoss.calculate()),
        expected_shape (the item shape this loss reads), loss_name (for
        the error message).
    Output: none.
    Side effects: none (host lengths only, no device sync).
    Exceptions: ValueError on a shape or item-count mismatch.
    """
    if item_embeddings.shape is not expected_shape:
        # e.g. InputShape.MULTI_CARD -> "multi-card"
        shape_text = expected_shape.name.lower().replace("_", "-")
        raise ValueError(
            f"{loss_name} expects {shape_text} item embeddings "
            f"(got {item_embeddings.shape.name})"
        )
    if len(item_embeddings) != len(identities):
        raise ValueError(
            f"item_embeddings ({len(item_embeddings)}) and identities "
            f"({len(identities)}) must be the same length"
        )


def require_card_aligned_identities(
    item_embeddings: PaddedEmbeddings,
    identities: Sequence[tuple[UUID, ...]],
    loss_name: str,
) -> None:
    """Each item's card count equals its identity's length, so card k of
    item i is identities[i][k] (ContrastiveBatch's positional contract).

    Shared by the losses that read positions: cross-slice (flat card k is
    flat identity k), odd one out (the intruder is the last card) and card
    in contexts (the anchor is the first). A mod that drops cards from an
    item but keeps its identity would silently misalign them.
    Inputs: item_embeddings, identities (already the same item count, see
        check_item_embeddings), loss_name (for the message).
    Output: none.
    Side effects: none (host lengths only).
    Exceptions: ValueError on any per-item mismatch.

    Example:
        >>> require_card_aligned_identities(item_embeddings, identities, "OddOneOutLoss")
    """
    identity_lengths = tuple(len(identity) for identity in identities)
    if item_embeddings.lengths != identity_lengths:
        raise ValueError(
            f"{loss_name} needs one identity per card: item lengths "
            f"{item_embeddings.lengths} != identity lengths {identity_lengths}"
        )


def _require_unit_count(units: torch.Tensor, masks: ComparisonMasks) -> None:
    """Private helper - single caller is info_nce_loss.
    Inputs: units, masks. Output: none. Side effects: none.
    Exceptions: ValueError unless units is (masks.unit_count, d).
    """
    if units.dim() != 2 or units.shape[0] != masks.unit_count:
        raise ValueError(
            f"units has shape {tuple(units.shape)}, want ({masks.unit_count}, d)"
        )


def _cosine_logits(units: torch.Tensor, temperature: float) -> torch.Tensor:
    """Private helper - single caller is info_nce_loss.
    Inputs: units (N, d), temperature.
    Output: (N, N) cosine similarity / temperature.
    Side effects: none beyond autograd. Exceptions: none.
    """
    normalized = torch.nn.functional.normalize(units, dim=1)
    return (normalized @ normalized.T) / temperature


def _negative_log_sum_exp(logits: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Each anchor's logsumexp over its valid negatives.

    Private helper - single caller is info_nce_loss.
    Inputs: logits (N, N), negative (N, N) bool on the same device.
    Output: (N,). A row with no negative holds a finite placeholder, never
        the result of a logsumexp over an all -inf row: the masked logits
        are filled with a finite value in such rows before the reduction
        (torch.where), and _pair_losses selects those rows away.
    Side effects: none beyond autograd. Exceptions: none.
    """
    has_negative = negative.any(dim=1, keepdim=True)
    masked = logits.masked_fill(~negative, float("-inf"))
    # Rows with no negative get finite zeros, so logsumexp never sees an
    # all -inf row; _pair_losses discards those rows
    finite = torch.where(has_negative, masked, torch.zeros_like(logits))
    return torch.logsumexp(finite, dim=1)


def _pair_losses(
    logits: torch.Tensor, negative_lse: torch.Tensor, has_negative: torch.Tensor
) -> torch.Tensor:
    """Every (anchor, candidate) pair's InfoNCE term.

    Private helper - single caller is info_nce_loss.
    Inputs: logits (N, N); negative_lse (N,), finite; has_negative (N,)
        bool.
    Output: (N, N): logaddexp(l[i, j], negative_lse[i]) - l[i, j] where
        anchor i has a negative, exactly 0 (torch.where, zero gradient)
        where it has none. Only positive pairs are read later.
    Side effects: none beyond autograd. Exceptions: none.
    """
    terms = torch.logaddexp(logits, negative_lse.unsqueeze(1)) - logits
    return torch.where(has_negative.unsqueeze(1), terms, torch.zeros_like(terms))


def _mean_over_anchors(
    pair_losses: torch.Tensor, positive: torch.Tensor
) -> torch.Tensor:
    """Mean over each anchor's positives, then over anchors with one.

    Private helper - single caller is info_nce_loss.
    Inputs: pair_losses (N, N), positive (N, N) bool on the same device.
    Output: scalar. Non-positive pairs are excluded with torch.where (not
        multiplication), so their values cannot leak NaN; the anchor count
        is clamped to 1 on the device, so no sync.
    Side effects: none beyond autograd. Exceptions: none.
    """
    # Each anchor's mean over its positives
    positive_sums = torch.where(
        positive, pair_losses, torch.zeros_like(pair_losses)
    ).sum(dim=1)
    positive_counts = positive.sum(dim=1)
    anchor_means = positive_sums / positive_counts.clamp(min=1)

    # Mean over the anchors that have a positive
    is_anchor = positive_counts > 0
    anchor_total = torch.where(
        is_anchor, anchor_means, torch.zeros_like(anchor_means)
    ).sum()
    return anchor_total / is_anchor.sum().clamp(min=1)


class ContrastiveLoss(Protocol):
    """Computes one scalar loss from every item's embedding in a batch,
    jointly - the batch-level counterpart to NocabLoss's row-independent
    shape. Owned privately by a ContrastiveDojo."""

    def calculate(
        self,
        item_embeddings: PaddedEmbeddings,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> torch.Tensor:
        """Compute this batch's contrastive loss.

        Inputs:
            item_embeddings: one row per item, same order/length as
                identities - shape (SINGLE_CARD vs. MULTI_CARD) is
                implementation-defined; a concrete ContrastiveLoss
                validates its own expected shape.
            identities: parallel to item_embeddings' rows - see
                ContrastiveBatch.identities.
            positive_cliques: index groups into items - see
                ContrastiveBatch.positive_cliques.
        Output: a scalar loss tensor.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined (expected: ValueError on a
            length mismatch between item_embeddings and identities, or
            on a shape this implementation doesn't support).
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
