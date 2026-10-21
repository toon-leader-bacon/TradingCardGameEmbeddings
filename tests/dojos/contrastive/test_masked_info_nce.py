"""Tests for the vectorized InfoNCE math (contrastive_loss.py): its
building blocks, and each InfoNCE style against the old per-anchor loop
(info_nce_oracle.py) on random batches."""

import math
import random
from typing import Callable, Hashable, Sequence
from uuid import UUID, uuid4

import pytest
import torch

from src.dojos.contrastive.contrastive_loss import (
    AnchorMasks,
    ContrastiveLoss,
    DegenerateBatchError,
    clique_keys,
    constant_logit_baseline,
    masked_info_nce_loss,
    mean_item_embeddings,
    pairwise_cosine_similarity,
    same_key_mask,
    unit_keys,
)
from src.dojos.contrastive.styles.cross_slice_card import CrossSliceCardInfoNCELoss
from src.dojos.contrastive.styles.missing_card import MissingCardInfoNCELoss
from src.dojos.contrastive.styles.single_card import SingleCardInfoNCELoss
from tests.dojos.contrastive import info_nce_oracle as oracle

CPU = torch.device("cpu")
WIDTH = 6
SEEDS = range(12)


def _masks(cliques: list[list[int]], identities: Sequence[Hashable]) -> AnchorMasks:
    positives = same_key_mask(clique_keys(cliques, len(identities), CPU))
    return AnchorMasks.of(positives, None, unit_keys(identities, CPU))


class TestKeys:
    def test_same_key_mask_compares_every_pair(self) -> None:
        assert same_key_mask(torch.tensor([0, 0, 1])).tolist() == [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ]

    @pytest.mark.parametrize(
        "keys", [torch.tensor([0.0, 1.0]), torch.tensor([[0, 1]])], ids=["float", "2d"]
    )
    def test_same_key_mask_rejects_keys_that_are_not_1d_int64(
        self, keys: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="1-D int64"):
            same_key_mask(keys)

    def test_unit_keys_number_units_in_first_seen_order(self) -> None:
        assert unit_keys(["a", "b", "a"], CPU).tolist() == [0, 1, 0]

    def test_clique_keys_give_a_unit_in_no_clique_its_own_key(self) -> None:
        assert clique_keys([[0, 2], [1]], 4, CPU).tolist() == [0, 1, 0, 5]

    def test_clique_keys_reject_a_unit_in_two_cliques(self) -> None:
        with pytest.raises(ValueError, match="more than one clique"):
            clique_keys([[0, 1], [1, 2]], 3, CPU)

    @pytest.mark.parametrize("index", [3, -1])
    def test_clique_keys_reject_an_index_out_of_range(self, index: int) -> None:
        with pytest.raises(ValueError, match="outside"):
            clique_keys([[0, index]], 3, CPU)


class TestAnchorMasks:
    def test_of_makes_everyone_else_a_negative(self) -> None:
        masks = _masks([[0, 1], [2, 3]], ["a", "b", "c", "d"])

        assert masks.negatives[0].tolist() == [False, False, True, True]
        assert masks.anchors.tolist() == [True] * 4

    def test_ignored_units_are_neither_positive_nor_negative(self) -> None:
        positives = same_key_mask(torch.tensor([0, 0, 1]))
        ignored = torch.zeros((3, 3), dtype=torch.bool)
        ignored[0, 2] = True

        masks = AnchorMasks.of(positives, ignored, unit_keys("abc", CPU))

        assert not masks.negatives[0, 2] and not masks.positives[0, 2]

    def test_a_positive_wins_over_ignored(self) -> None:
        positives = same_key_mask(torch.tensor([0, 0]))
        everything = torch.ones((2, 2), dtype=torch.bool)

        masks = AnchorMasks.of(positives, everything, unit_keys("ab", CPU))

        assert masks.positives[0, 1] and not masks.negatives.any()

    def test_of_rejects_identity_keys_of_another_length(self) -> None:
        positives = same_key_mask(torch.tensor([0, 0]))

        with pytest.raises(ValueError, match="disagree"):
            AnchorMasks.of(positives, None, unit_keys("abc", CPU))

    def test_rejects_a_self_pair(self) -> None:
        eye = torch.eye(2, dtype=torch.bool)

        with pytest.raises(ValueError, match="diagonal"):
            AnchorMasks(eye, torch.zeros((2, 2), dtype=torch.bool))

    def test_rejects_a_unit_both_positive_and_negative(self) -> None:
        both = ~torch.eye(2, dtype=torch.bool)

        with pytest.raises(ValueError, match="disjoint"):
            AnchorMasks(both, both)

    def test_rejects_masks_that_are_not_square_bool(self) -> None:
        with pytest.raises(ValueError, match="square bool"):
            AnchorMasks(torch.zeros((2, 2)), torch.zeros((2, 2)))


class TestMaskedInfoNCELoss:
    def test_an_anchor_with_no_negative_scores_zero_with_a_finite_gradient(
        self,
    ) -> None:
        # One deck of two: each anchor's only other unit is its positive
        embeddings = torch.randn(2, WIDTH, requires_grad=True)
        masks = _masks([[0, 1]], ["a", "b"])

        loss = masked_info_nce_loss(
            pairwise_cosine_similarity(list(embeddings)), masks, 0.07
        )
        loss.backward()

        assert loss.item() == 0.0
        assert embeddings.grad is not None
        assert torch.isfinite(embeddings.grad).all()

    def test_raises_when_no_unit_has_a_positive(self) -> None:
        masks = _masks([[0], [1]], ["a", "b"])

        with pytest.raises(DegenerateBatchError, match="positive"):
            masked_info_nce_loss(torch.zeros((2, 2)), masks, 0.07)

    def test_rejects_a_similarity_of_another_shape(self) -> None:
        masks = _masks([[0, 1]], ["a", "b"])

        with pytest.raises(ValueError, match="similarity"):
            masked_info_nce_loss(torch.zeros((3, 3)), masks, 0.07)

    def test_a_half_precision_similarity_gives_a_float32_loss(self) -> None:
        similarity = pairwise_cosine_similarity(list(torch.randn(6, WIDTH)))
        masks = _masks([[0, 1], [2, 3], [4, 5]], "abcdef")

        full = masked_info_nce_loss(similarity, masks, 0.07)
        half = masked_info_nce_loss(similarity.half(), masks, 0.07)

        assert half.dtype == torch.float32
        assert half.item() == pytest.approx(full.item(), rel=1e-2)


def test_an_anchor_without_negatives_beside_one_with_them_matches_the_loop() -> None:
    # Item 2 duplicates item 0: anchor 0 has no negative left, anchor 1 has one
    identities = ["a", "b", "a"]
    cliques = [[0, 1], [2]]
    embeddings = torch.randn(3, WIDTH, dtype=torch.float64)
    similarity = pairwise_cosine_similarity(list(embeddings))

    result = masked_info_nce_loss(similarity, _masks(cliques, identities), 0.07)

    expected = oracle.oracle_loss(
        similarity, identities, oracle.single_card_relations(cliques), 0.07
    )
    assert result.item() == pytest.approx(expected.item(), rel=1e-5)


def test_calculate_raises_on_a_degenerate_batch() -> None:
    with pytest.raises(DegenerateBatchError):
        SingleCardInfoNCELoss().calculate(
            [torch.randn(WIDTH), torch.randn(WIDTH)],
            [(uuid4(),), (uuid4(),)],
            [[0], [1]],
        )


class TestConstantLogitBaseline:
    def test_three_decks_of_two_give_ln_5(self) -> None:
        masks = _masks([[0, 1], [2, 3], [4, 5]], "abcdef")

        assert constant_logit_baseline(masks) == pytest.approx(math.log(5))

    def test_raises_when_no_anchor_has_a_negative(self) -> None:
        with pytest.raises(DegenerateBatchError, match="negative"):
            constant_logit_baseline(_masks([[0, 1]], ["a", "b"]))


# --- Equivalence with the old per-anchor loop -----------------------------


def _uuid_pool(size: int) -> list[UUID]:
    # A small pool, so random draws repeat (identity duplicates)
    return [uuid4() for _ in range(size)]


def _single_card_batch(
    rng: random.Random,
) -> tuple[list[tuple[UUID, ...]], list[list[int]]]:
    pool = _uuid_pool(6)
    identities: list[tuple[UUID, ...]] = []
    cliques: list[list[int]] = []
    for deck in range(rng.randint(2, 5)):
        start = len(identities)
        # Size 1 to 4: singletons and multi-positive cliques; the first
        # deck has at least 2, for the duplicate below
        for _ in range(rng.randint(2 if deck == 0 else 1, 4)):
            identities.append((rng.choice(pool),))
        cliques.append(list(range(start, len(identities))))
    # Always one in-clique duplicate: it must stay a positive
    identities[cliques[0][1]] = identities[cliques[0][0]]
    return identities, cliques


def _cross_slice_batch(
    rng: random.Random,
) -> tuple[list[tuple[UUID, ...]], list[list[int]]]:
    pool = _uuid_pool(8)
    cards_per_item = rng.randint(2, 3)
    identities: list[tuple[UUID, ...]] = []
    cliques: list[list[int]] = []
    for _ in range(rng.randint(2, 4)):
        start = len(identities)
        for _ in range(rng.randint(1, 3)):
            identities.append(tuple(rng.choice(pool) for _ in range(cards_per_item)))
        cliques.append(list(range(start, len(identities))))
    return identities, cliques


def _missing_card_batch(
    rng: random.Random,
) -> tuple[list[tuple[UUID, ...]], list[list[int]]]:
    contexts = _uuid_pool(10)
    # Three possible missing cards: twin decks are common
    missing = _uuid_pool(3)
    identities: list[tuple[UUID, ...]] = []
    cliques: list[list[int]] = []
    for _ in range(rng.randint(2, 6)):
        start = len(identities)
        identities.append(tuple(rng.choice(contexts) for _ in range(rng.randint(2, 3))))
        identities.append((rng.choice(missing),))
        cliques.append([start, start + 1])
    return identities, cliques


def _single_card_oracle(
    embeddings: list[torch.Tensor], identities: list, cliques: list[list[int]]
) -> tuple[torch.Tensor, list[Hashable], oracle.Relations]:
    similarity = pairwise_cosine_similarity(embeddings)
    return similarity, identities, oracle.single_card_relations(cliques)


def _cross_slice_oracle(
    items: list[list[torch.Tensor]], identities: list, cliques: list[list[int]]
) -> tuple[torch.Tensor, list[Hashable], oracle.Relations]:
    flat_identities, item_of_card = oracle.flat_card_identities(identities)
    similarity = pairwise_cosine_similarity([card for item in items for card in item])
    relations = oracle.cross_slice_relations(item_of_card, cliques)
    return similarity, list(flat_identities), relations


def _missing_card_oracle(
    items: list[list[torch.Tensor]], identities: list, cliques: list[list[int]]
) -> tuple[torch.Tensor, list[Hashable], oracle.Relations]:
    similarity = pairwise_cosine_similarity(mean_item_embeddings(items))
    return similarity, identities, oracle.missing_card_relations(cliques, identities)


def _single_card_items(leaf: torch.Tensor, identities: list) -> list:
    return list(leaf[: len(identities)])


def _multi_card_items(leaf: torch.Tensor, identities: list) -> list:
    items: list[list[torch.Tensor]] = []
    row = 0
    for identity in identities:
        items.append(list(leaf[row : row + len(identity)]))
        row += len(identity)
    return items


StyleCase = tuple[
    Callable[[], ContrastiveLoss],
    Callable[[random.Random], tuple[list, list[list[int]]]],
    Callable[[torch.Tensor, list], list],
    Callable[..., tuple[torch.Tensor, list[Hashable], oracle.Relations]],
]
STYLE_CASES: dict[str, StyleCase] = {
    "single_card": (
        SingleCardInfoNCELoss,
        _single_card_batch,
        _single_card_items,
        _single_card_oracle,
    ),
    "cross_slice": (
        CrossSliceCardInfoNCELoss,
        _cross_slice_batch,
        _multi_card_items,
        _cross_slice_oracle,
    ),
    "missing_card": (
        MissingCardInfoNCELoss,
        _missing_card_batch,
        _multi_card_items,
        _missing_card_oracle,
    ),
}


@pytest.mark.parametrize("style", STYLE_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_matches_the_per_anchor_loop(style: str, seed: int) -> None:
    make_loss, make_batch, make_items, oracle_inputs = STYLE_CASES[style]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    identities, cliques = make_batch(rng)
    card_count = sum(len(identity) for identity in identities)
    loss_fn = make_loss()

    # The old loop in float64 (exact; the new loss computes in float32),
    # on its own copy of the same embeddings
    old_leaf = torch.randn(card_count, WIDTH, dtype=torch.float64).requires_grad_()
    similarity, unit_identities, relations = oracle_inputs(
        make_items(old_leaf, identities), identities, cliques
    )
    try:
        expected = oracle.oracle_loss(similarity, unit_identities, relations, 0.07)
        expected_baseline = oracle.oracle_baseline(unit_identities, relations)
    except DegenerateBatchError:
        with pytest.raises(DegenerateBatchError):
            loss_fn.constant_logit_loss(identities, cliques)
        return
    expected.backward()

    # The vectorized loss on the same values
    new_leaf = old_leaf.detach().clone().requires_grad_()
    result = loss_fn.calculate(make_items(new_leaf, identities), identities, cliques)
    result.backward()

    assert result.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
    assert old_leaf.grad is not None and new_leaf.grad is not None
    torch.testing.assert_close(new_leaf.grad, old_leaf.grad, rtol=1e-4, atol=1e-6)
    assert loss_fn.constant_logit_loss(identities, cliques) == pytest.approx(
        expected_baseline, rel=1e-12
    )
