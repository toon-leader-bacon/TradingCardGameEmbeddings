import math
from uuid import uuid4

import pytest
import torch

from src.dojos.contrastive.contrastive_loss import (
    SingleCardInfoNCELoss,
    _anchor_loss,
    _pairwise_cosine_similarity,
    _valid_negative_mask,
)


class TestInit:
    def test_raises_on_non_positive_temperature(self) -> None:
        with pytest.raises(ValueError):
            SingleCardInfoNCELoss(temperature=0.0)


class TestPairwiseCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self) -> None:
        embeddings = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])]

        similarity = _pairwise_cosine_similarity(embeddings)

        assert similarity[0, 1].item() == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self) -> None:
        embeddings = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

        similarity = _pairwise_cosine_similarity(embeddings)

        assert similarity[0, 1].item() == pytest.approx(0.0)

    def test_diagonal_is_self_similarity_of_one(self) -> None:
        embeddings = [torch.randn(4) for _ in range(3)]

        similarity = _pairwise_cosine_similarity(embeddings)

        assert torch.allclose(torch.diagonal(similarity), torch.ones(3))


class TestValidNegativeMask:
    def test_excludes_the_diagonal(self) -> None:
        identities = [(uuid4(),), (uuid4(),), (uuid4(),)]

        mask = _valid_negative_mask(identities)

        assert not mask.diagonal().any()

    def test_excludes_an_exact_identity_duplicate(self) -> None:
        duplicate_uuid = uuid4()
        identities = [(duplicate_uuid,), (uuid4(),), (duplicate_uuid,)]

        mask = _valid_negative_mask(identities)

        assert not mask[0, 2]
        assert not mask[2, 0]

    def test_allows_a_non_duplicate_pair(self) -> None:
        identities = [(uuid4(),), (uuid4(),)]

        mask = _valid_negative_mask(identities)

        assert mask[0, 1]
        assert mask[1, 0]


class TestPositivesByAnchor:
    def test_expands_each_group_into_mutual_positives(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        positives_by_anchor = loss_fn._positives_by_anchor([[0, 1, 2]])

        assert positives_by_anchor == {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    def test_omits_a_singleton_group_from_the_mapping(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        positives_by_anchor = loss_fn._positives_by_anchor([[0], [1, 2]])

        assert 0 not in positives_by_anchor
        assert positives_by_anchor == {1: [2], 2: [1]}


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
        # see TestAnchorLoss below for a case that actually
        # distinguishes the per-positive averaging math from a bug that
        # only used one of an anchor's positives.
        loss_fn = SingleCardInfoNCELoss(temperature=1.0)
        item_embeddings = list(torch.eye(4))
        identities = [(uuid4(),) for _ in range(4)]
        positive_cliques = [[0, 1, 2]]

        result = loss_fn.calculate(item_embeddings, identities, positive_cliques)

        assert result.item() >= 0
        assert math.isfinite(result.item())


class TestAnchorLoss:
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
        valid_negative_mask = torch.tensor(
            [
                [False, True, True, True],
                [True, False, True, True],
                [True, True, False, True],
                [True, True, True, False],
            ]
        )

        result = _anchor_loss(0, [1, 2], set(), similarity, valid_negative_mask, 1.0)

        # positive=1: candidates' similarities [2.0, 0.0] (vs. negative 3)
        loss_for_positive_1 = math.log(math.exp(2.0) + math.exp(0.0)) - 2.0
        # positive=2: candidates' similarities [0.0, 0.0] (vs. negative 3)
        loss_for_positive_2 = math.log(2.0)
        expected = (loss_for_positive_1 + loss_for_positive_2) / 2
        assert result.item() == pytest.approx(expected, rel=1e-5)
