"""The per-anchor InfoNCE loop the vectorized loss replaced, kept as a
test oracle (plans/vectorized_contrastive_loss.md).

The four math functions are the old contrastive_loss.py ones, and the
three relation builders are the old styles' private dict expansions,
both unchanged apart from becoming module functions. The loop reads mask
entries one at a time, so it is slow; keep the batches small. Test code
only: nothing in src/ imports this.
"""

import math
from typing import Hashable, Mapping, Sequence
from uuid import UUID

import torch

from src.dojos.contrastive.contrastive_loss import DegenerateBatchError

# anchor -> its positives, anchor -> its extra exclusions
Relations = tuple[dict[int, list[int]], dict[int, set[int]]]


def identity_negative_mask(
    identities: Sequence[Hashable], device: torch.device
) -> torch.Tensor:
    """(N, N) bool: True where j may be a negative for i (not i itself,
    not an exact-identity duplicate)."""
    unit_count = len(identities)
    mask = torch.ones((unit_count, unit_count), dtype=torch.bool)
    for i in range(unit_count):
        mask[i, i] = False
        for j in range(i + 1, unit_count):
            if identities[i] == identities[j]:
                mask[i, j] = False
                mask[j, i] = False
    return mask.to(device)


def anchor_loss(
    anchor_index: int,
    anchor_positives: list[int],
    excluded_indices: set[int],
    similarity: torch.Tensor,
    valid_negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """One anchor's InfoNCE loss, averaged over its positives."""
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
        candidate_indices = [positive_index] + valid_negative_indices
        candidate_similarities = (
            similarity[anchor_index, candidate_indices] / temperature
        )
        log_probabilities = torch.log_softmax(candidate_similarities, dim=0)
        per_positive_losses.append(-log_probabilities[0])

    return torch.stack(per_positive_losses).mean()


def mean_anchor_loss(
    positives_by_anchor: Mapping[int, list[int]],
    excluded_by_anchor: Mapping[int, set[int]],
    similarity: torch.Tensor,
    valid_negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """anchor_loss for every anchor, averaged."""
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
    """mean over anchors of ln(1 + its negative count)."""
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


def single_card_relations(positive_cliques: list[list[int]]) -> Relations:
    """SingleCardInfoNCELoss's old _positives_by_anchor: every clique
    member is a positive of every other; nothing is excluded."""
    result: dict[int, list[int]] = {}
    for group in positive_cliques:
        for index in group:
            other_indices = [other for other in group if other != index]
            if other_indices:
                result[index] = other_indices
    return result, {}


def flat_card_identities(
    identities: list[tuple[UUID, ...]],
) -> tuple[list[UUID], list[int]]:
    """(one uuid per card, each card's item index), item by item."""
    flat_identities: list[UUID] = []
    item_of_card: list[int] = []
    for item_index, item_identities in enumerate(identities):
        for card_identity in item_identities:
            flat_identities.append(card_identity)
            item_of_card.append(item_index)
    return flat_identities, item_of_card


def cross_slice_relations(
    item_of_card: list[int], positive_cliques: list[list[int]]
) -> Relations:
    """CrossSliceCardInfoNCELoss's old _positives_and_exclusions_by_anchor:
    a card's positives are the other items' cards in its clique; its own
    item's other cards are excluded."""
    cards_of_item: dict[int, list[int]] = {}
    for card_index, item_index in enumerate(item_of_card):
        cards_of_item.setdefault(item_index, []).append(card_index)
    positives_by_anchor: dict[int, list[int]] = {}
    excluded_by_anchor: dict[int, set[int]] = {}
    for clique in positive_cliques:
        for item_index in clique:
            own_cards = cards_of_item.get(item_index, [])
            other_cards = [
                card
                for other_item in clique
                if other_item != item_index
                for card in cards_of_item.get(other_item, [])
            ]
            if not other_cards:
                continue
            for card in own_cards:
                positives_by_anchor[card] = other_cards
                excluded_by_anchor[card] = {
                    other for other in own_cards if other != card
                }
    return positives_by_anchor, excluded_by_anchor


def missing_card_relations(
    positive_cliques: list[list[int]], identities: list[tuple[UUID, ...]]
) -> Relations:
    """MissingCardInfoNCELoss's old _positives_and_exclusions, over
    [context, card] cliques: each unit's positive is its partner; it
    excludes the other same-role units and the other role's units of
    every other deck missing the same card."""
    pairs = [(clique[0], clique[1]) for clique in positive_cliques]
    positives_by_anchor: dict[int, list[int]] = {}
    excluded_by_anchor: dict[int, set[int]] = {}
    context_indices = {context for context, _ in pairs}
    card_indices = {card for _, card in pairs}

    pairs_by_missing_card: dict[tuple[UUID, ...], list[tuple[int, int]]] = {}
    for pair in pairs:
        pairs_by_missing_card.setdefault(identities[pair[1]], []).append(pair)

    for pair in pairs:
        context, card = pair
        twins = pairs_by_missing_card[identities[card]]
        positives_by_anchor[context] = [card]
        excluded_by_anchor[context] = (context_indices - {context}) | {
            twin[1] for twin in twins if twin != pair
        }
        positives_by_anchor[card] = [context]
        excluded_by_anchor[card] = (card_indices - {card}) | {
            twin[0] for twin in twins if twin != pair
        }
    return positives_by_anchor, excluded_by_anchor


def oracle_loss(
    similarity: torch.Tensor,
    unit_identities: Sequence[Hashable],
    relations: Relations,
    temperature: float,
) -> torch.Tensor:
    """The old loss over one batch's comparison units."""
    positives_by_anchor, excluded_by_anchor = relations
    mask = identity_negative_mask(unit_identities, similarity.device)
    return mean_anchor_loss(
        positives_by_anchor, excluded_by_anchor, similarity, mask, temperature
    )


def oracle_baseline(unit_identities: Sequence[Hashable], relations: Relations) -> float:
    """The old constant-logit baseline over one batch's comparison units."""
    positives_by_anchor, excluded_by_anchor = relations
    return mean_constant_logit_loss(
        unit_identities, positives_by_anchor, excluded_by_anchor
    )
