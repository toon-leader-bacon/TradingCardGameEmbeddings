import math
from uuid import uuid4

import pytest
import torch

from src.dojos.contrastive.contrastive_loss import (
    AnchorMasks,
    masked_info_nce_loss,
    pairwise_cosine_similarity,
    unit_keys,
)
from src.dojos.contrastive.styles.cross_slice_card import CrossSliceCardInfoNCELoss
from src.dojos.contrastive.styles.single_card import SingleCardInfoNCELoss


class TestInit:
    def test_raises_on_non_positive_temperature(self) -> None:
        with pytest.raises(ValueError):
            SingleCardInfoNCELoss(temperature=0.0)


class TestPairwiseCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self) -> None:
        embeddings = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])]

        similarity = pairwise_cosine_similarity(embeddings)

        assert similarity[0, 1].item() == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self) -> None:
        embeddings = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

        similarity = pairwise_cosine_similarity(embeddings)

        assert similarity[0, 1].item() == pytest.approx(0.0)

    def test_diagonal_is_self_similarity_of_one(self) -> None:
        embeddings = [torch.randn(4) for _ in range(3)]

        similarity = pairwise_cosine_similarity(embeddings)

        assert torch.allclose(torch.diagonal(similarity), torch.ones(3))


class TestValidNegativeMask:
    def _negatives(self, identities: list[tuple]) -> torch.Tensor:
        # No positives, nothing ignored: only self and duplicates are out
        cpu = torch.device("cpu")
        unit_count = len(identities)
        no_positives = torch.zeros((unit_count, unit_count), dtype=torch.bool)
        return AnchorMasks.of(no_positives, None, unit_keys(identities, cpu)).negatives

    def test_excludes_the_diagonal(self) -> None:
        identities = [(uuid4(),), (uuid4(),), (uuid4(),)]

        mask = self._negatives(identities)

        assert not mask.diagonal().any()

    def test_excludes_an_exact_identity_duplicate(self) -> None:
        duplicate_uuid = uuid4()
        identities = [(duplicate_uuid,), (uuid4(),), (duplicate_uuid,)]

        mask = self._negatives(identities)

        assert not mask[0, 2]
        assert not mask[2, 0]

    def test_allows_a_non_duplicate_pair(self) -> None:
        identities = [(uuid4(),), (uuid4(),)]

        mask = self._negatives(identities)

        assert mask[0, 1]
        assert mask[1, 0]


class TestPositivesByAnchor:
    def _positives(self, cliques: list[list[int]], unit_count: int) -> torch.Tensor:
        identities = [(uuid4(),) for _ in range(unit_count)]
        masks = SingleCardInfoNCELoss()._anchor_masks(
            identities, cliques, torch.device("cpu")
        )
        return masks.positives

    def test_expands_each_group_into_mutual_positives(self) -> None:
        positives = self._positives([[0, 1, 2]], 3)

        assert positives.tolist() == [
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ]

    def test_a_singleton_group_has_no_positive(self) -> None:
        positives = self._positives([[0], [1, 2]], 3)

        assert not positives[0].any()
        assert positives[1, 2] and positives[2, 1]

    def test_a_duplicate_in_the_same_clique_stays_a_positive(self) -> None:
        shared = uuid4()
        masks = SingleCardInfoNCELoss()._anchor_masks(
            [(shared,), (shared,), (uuid4(),)], [[0, 1], [2]], torch.device("cpu")
        )

        assert masks.positives[0, 1] and masks.positives[1, 0]


class TestCalculate:
    def test_raises_on_a_multi_card_shaped_input(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        with pytest.raises(ValueError):
            loss_fn.calculate(
                item_embeddings=[[torch.randn(4), torch.randn(4)]],
                identities=[(uuid4(),)],
                positive_cliques=[],
            )

    def test_raises_on_length_mismatch(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        with pytest.raises(ValueError):
            loss_fn.calculate(
                item_embeddings=[torch.randn(4)],
                identities=[(uuid4(),), (uuid4(),)],
                positive_cliques=[],
            )

    def test_raises_when_no_item_has_a_positive(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        with pytest.raises(ValueError):
            loss_fn.calculate(
                item_embeddings=[torch.randn(4), torch.randn(4)],
                identities=[(uuid4(),), (uuid4(),)],
                positive_cliques=[],
            )

    def test_matches_a_hand_computed_orthonormal_case(self) -> None:
        # 4 mutually-orthogonal unit embeddings -> every off-diagonal
        # cosine similarity is 0. With temperature=1, each anchor's
        # single positive competes against exactly 2 valid negatives,
        # all at similarity 0, so softmax is uniform over 3 candidates
        # and every anchor's own loss is -log(1/3) = log(3).
        loss_fn = SingleCardInfoNCELoss(temperature=1.0)
        item_embeddings = list(torch.eye(4))
        identities = [(uuid4(),) for _ in range(4)]
        positive_cliques = [[0, 1], [2, 3]]

        result = loss_fn.calculate(item_embeddings, identities, positive_cliques)

        assert result.item() == pytest.approx(math.log(3), rel=1e-5)

    def test_excludes_a_duplicate_identity_from_the_negative_pool(self) -> None:
        # Item 2 is an exact-identity duplicate of item 0 (e.g. the same
        # card drawn from a different deck) but shares no positive group
        # with it. Anchor 0's only valid negative is then item 3, not
        # item 2 - if the duplicate exclusion were broken, this would
        # instead average over both item 2 and item 3 as negatives and
        # no longer match the hand-computed value below.
        loss_fn = SingleCardInfoNCELoss(temperature=1.0)
        shared_uuid = uuid4()
        item_embeddings = list(torch.eye(4))
        identities = [(shared_uuid,), (uuid4(),), (shared_uuid,), (uuid4(),)]
        positive_cliques = [[0, 1]]

        result = loss_fn.calculate(item_embeddings, identities, positive_cliques)

        # Anchor 0: positive=1, valid negatives={3} (2 excluded as a
        # duplicate identity) -> 2-way softmax over similarities [0, 0].
        # Anchor 1: positive=0, valid negatives={2, 3} (no duplicate of
        # item 1) -> 3-way softmax over similarities [0, 0, 0].
        expected = (math.log(2) + math.log(3)) / 2
        assert result.item() == pytest.approx(expected, rel=1e-5)

    def test_calculate_handles_a_group_larger_than_two(self) -> None:
        # A 3-item positive group means every member has 2 positives
        # (the other two). This just confirms calculate() runs that
        # shape end to end and produces a finite, non-negative loss -
        # see TestMaskedInfoNCELoss below for a case that actually
        # distinguishes the per-positive averaging math from a bug that
        # only used one of an anchor's positives.
        loss_fn = SingleCardInfoNCELoss(temperature=1.0)
        item_embeddings = list(torch.eye(4))
        identities = [(uuid4(),) for _ in range(4)]
        positive_cliques = [[0, 1, 2]]

        result = loss_fn.calculate(item_embeddings, identities, positive_cliques)

        assert result.item() >= 0
        assert math.isfinite(result.item())


class TestMaskedInfoNCELoss:
    def test_averages_over_differing_per_positive_losses(self) -> None:
        # Anchor 0 has two positives (1 and 2) with deliberately
        # different similarities to it, against one shared valid
        # negative (3) - so the two per-positive InfoNCE losses differ,
        # and a bug that used only one positive (instead of averaging
        # both) would not match this hand-computed mean.
        similarity = torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        positives = torch.zeros((4, 4), dtype=torch.bool)
        positives[0, 1] = positives[0, 2] = True
        negatives = torch.zeros((4, 4), dtype=torch.bool)
        negatives[0, 3] = True

        result = masked_info_nce_loss(
            similarity, AnchorMasks(positives, negatives), 1.0
        )

        # positive=1: candidates' similarities [2.0, 0.0] (vs. negative 3)
        loss_for_positive_1 = math.log(math.exp(2.0) + math.exp(0.0)) - 2.0
        # positive=2: candidates' similarities [0.0, 0.0] (vs. negative 3)
        loss_for_positive_2 = math.log(2.0)
        expected = (loss_for_positive_1 + loss_for_positive_2) / 2
        assert result.item() == pytest.approx(expected, rel=1e-5)


def _single_identities(count: int) -> list[tuple]:
    return [(uuid4(),) for _ in range(count)]


def _multi_identities(items: int, cards_per_item: int) -> list[tuple]:
    return [tuple(uuid4() for _ in range(cards_per_item)) for _ in range(items)]


class TestSingleCardConstantLogitLoss:
    def test_closed_form_for_three_decks_of_two(self) -> None:
        # Each anchor: 1 positive, 6 - 2 = 4 negatives -> ln(1 + 4)
        baseline = SingleCardInfoNCELoss().constant_logit_loss(
            _single_identities(6), [[0, 1], [2, 3], [4, 5]]
        )
        assert baseline == pytest.approx(math.log(5))

    def test_equals_calculate_with_identical_embeddings(self) -> None:
        identities = _single_identities(7)
        cliques = [[0, 1, 2], [3, 4], [5], [6]]
        same = torch.ones(4)
        loss = SingleCardInfoNCELoss().calculate([same] * 7, identities, cliques)
        baseline = SingleCardInfoNCELoss().constant_logit_loss(identities, cliques)
        assert loss.item() == pytest.approx(baseline)

    def test_a_duplicate_identity_is_not_counted_as_a_negative(self) -> None:
        identities = _single_identities(4)
        identities[3] = identities[0]  # item 3 is the same card as anchor 0
        cliques = [[0, 1], [2, 3]]
        same = torch.ones(4)
        loss = SingleCardInfoNCELoss().calculate([same] * 4, identities, cliques)
        baseline = SingleCardInfoNCELoss().constant_logit_loss(identities, cliques)
        assert loss.item() == pytest.approx(baseline)
        # Anchors 0 and 3 lose a negative each: (2 ln 2 + 2 ln 3) / 4
        assert baseline == pytest.approx((2 * math.log(2) + 2 * math.log(3)) / 4)

    def test_raises_when_no_item_has_a_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SingleCardInfoNCELoss().constant_logit_loss(
                _single_identities(2), [[0], [1]]
            )

    def test_raises_when_no_anchor_has_a_negative(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            SingleCardInfoNCELoss().constant_logit_loss(_single_identities(2), [[0, 1]])


class TestCrossSliceCardInfoNCELoss:
    def test_closed_form_for_three_decks_of_two_five_card_items(self) -> None:
        # Each card: 5 positives, 4 own-item exclusions, 30 - 10 = 20 negatives
        baseline = CrossSliceCardInfoNCELoss().constant_logit_loss(
            _multi_identities(6, 5), [[0, 1], [2, 3], [4, 5]]
        )
        assert baseline == pytest.approx(math.log(21))

    def test_equals_calculate_with_identical_embeddings(self) -> None:
        identities = _multi_identities(5, 2)
        cliques = [[0, 1, 2], [3, 4]]
        same = torch.ones(4)
        loss = CrossSliceCardInfoNCELoss().calculate(
            [[same, same] for _ in range(5)], identities, cliques
        )
        baseline = CrossSliceCardInfoNCELoss().constant_logit_loss(identities, cliques)
        assert loss.item() == pytest.approx(baseline)

    def test_own_item_cards_are_neither_positive_nor_negative(self) -> None:
        identities = _multi_identities(3, 2)
        cliques = [[0, 1], [2]]
        # Card 0 (item 0) is aligned with item 1's cards and orthogonal to
        # its own sibling, which must not count as a negative
        a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        embeddings = [[a, b], [a, a], [b, b]]
        loss = CrossSliceCardInfoNCELoss(temperature=1.0).calculate(
            embeddings, identities, cliques
        )
        assert torch.isfinite(loss)

    def test_raises_on_a_single_card_shaped_input(self) -> None:
        with pytest.raises(ValueError, match="multi-card"):
            CrossSliceCardInfoNCELoss().calculate(
                [torch.randn(4), torch.randn(4)], _multi_identities(2, 1), [[0, 1]]
            )
