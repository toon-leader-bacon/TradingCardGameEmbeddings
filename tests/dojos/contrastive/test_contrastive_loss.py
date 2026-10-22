import math
from typing import List, Sequence
from uuid import uuid4

import pytest
import torch

from src.dojos.contrastive.contrastive_loss import (
    ComparisonMasks,
    DegenerateBatchError,
    clique_positive_mask,
    constant_logit_baseline,
    info_nce_loss,
    info_nce_masks,
    same_key_mask,
)
from src.dojos.contrastive.styles.cross_slice_card import CrossSliceCardInfoNCELoss
from src.dojos.contrastive.styles.single_card import SingleCardInfoNCELoss
from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape


def _singles(embeddings: Sequence[torch.Tensor]) -> PaddedEmbeddings:
    return PaddedEmbeddings.from_single_cards(torch.stack(list(embeddings)))


def _multis(items: List[List[torch.Tensor]]) -> PaddedEmbeddings:
    return PaddedEmbeddings.from_nested(
        items, InputShape.MULTI_CARD, items[0][0].shape[0]
    )


def _reference_info_nce(
    units: torch.Tensor, masks: ComparisonMasks, temperature: float
) -> torch.Tensor:
    """The per-anchor loop the vectorized loss replaced (an oracle): for
    each anchor with a positive, -log_softmax over [positive, negatives],
    averaged over its positives, then over anchors."""
    normalized = torch.nn.functional.normalize(units, dim=1)
    logits = normalized @ normalized.T / temperature
    per_anchor = []
    for anchor in range(units.shape[0]):
        positives = masks.positive[anchor].nonzero().flatten().tolist()
        if not positives:
            continue
        negatives = masks.negative[anchor].nonzero().flatten().tolist()
        per_positive = [
            -torch.log_softmax(logits[anchor, [positive] + negatives], dim=0)[0]
            for positive in positives
        ]
        per_anchor.append(torch.stack(per_positive).mean())
    return torch.stack(per_anchor).mean()


class TestInit:
    def test_raises_on_non_positive_temperature(self) -> None:
        with pytest.raises(ValueError):
            SingleCardInfoNCELoss(temperature=0.0)


class TestSameKeyMask:
    def test_marks_equal_keys_including_the_diagonal(self) -> None:
        mask = same_key_mask(["a", "b", "a"])

        assert mask.tolist() == [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ]

    def test_compares_uuid_tuples_by_value(self) -> None:
        shared = uuid4()

        mask = same_key_mask([(shared,), (uuid4(),), (shared,)])

        assert mask[0, 2] and mask[2, 0] and not mask[0, 1]

    def test_is_empty_for_no_keys(self) -> None:
        assert same_key_mask([]).shape == (0, 0)


class TestCliquePositiveMask:
    def test_expands_each_clique_into_mutual_positives(self) -> None:
        mask = clique_positive_mask([[0, 1, 2]], 3)

        assert mask.tolist() == [
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ]

    def test_a_singleton_or_missing_index_has_no_positive(self) -> None:
        mask = clique_positive_mask([[0], [1, 2]], 4)

        assert not mask[0].any() and not mask[3].any()
        assert mask[1, 2] and mask[2, 1]

    def test_raises_on_an_out_of_range_index(self) -> None:
        with pytest.raises(ValueError, match="range"):
            clique_positive_mask([[0, 3]], 3)

    def test_raises_on_an_index_in_two_cliques(self) -> None:
        with pytest.raises(ValueError, match="more than one"):
            clique_positive_mask([[0, 1], [1, 2]], 3)


class TestComparisonMasks:
    def test_info_nce_masks_excludes_self_positives_duplicates_and_exclusions(
        self,
    ) -> None:
        shared = uuid4()
        identities = [(shared,), (uuid4(),), (shared,), (uuid4(),)]
        excluded = torch.zeros(4, 4, dtype=torch.bool)
        excluded[1, 3] = True

        masks = info_nce_masks(clique_positive_mask([[0, 1]], 4), identities, excluded)

        assert masks.negative.tolist() == [
            [False, False, False, True],  # 1 positive, 2 duplicate
            [False, False, True, False],  # 0 positive, 3 excluded
            [False, True, False, True],  # 0 duplicate
            [True, True, True, False],
        ]

    def test_rejects_overlapping_masks(self) -> None:
        positive = clique_positive_mask([[0, 1]], 2)
        with pytest.raises(ValueError, match="both"):
            ComparisonMasks(positive, positive.clone())

    def test_rejects_a_device_mask(self) -> None:
        mask = torch.zeros(2, 2, dtype=torch.bool)
        with pytest.raises(ValueError, match="CPU"):
            ComparisonMasks(mask, mask.to("meta"))

    def test_require_some_anchor_raises_without_a_positive(self) -> None:
        empty = torch.zeros(2, 2, dtype=torch.bool)
        with pytest.raises(DegenerateBatchError):
            ComparisonMasks(
                empty, ~torch.eye(2, dtype=torch.bool)
            ).require_some_anchor()


class TestInfoNceLoss:
    def test_averages_over_differing_per_positive_losses(self) -> None:
        # Anchor 0 has two positives (1 and 2) at different similarities,
        # against one shared negative (3): a bug that used only one
        # positive would not match this hand-computed mean
        units = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        positive = torch.zeros(4, 4, dtype=torch.bool)
        positive[0, 1] = positive[0, 2] = True
        negative = torch.zeros(4, 4, dtype=torch.bool)
        negative[0, 3] = True

        result = info_nce_loss(units, ComparisonMasks(positive, negative), 1.0)

        loss_for_positive_1 = math.log(math.exp(1.0) + math.exp(0.0)) - 1.0
        loss_for_positive_2 = math.log(2.0)
        expected = (loss_for_positive_1 + loss_for_positive_2) / 2
        assert result.item() == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("seed", range(5))
    def test_equals_the_per_anchor_reference_on_random_masks(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        count = 9
        units = torch.randn(count, 6, generator=generator)
        cliques = [[0, 1, 2], [3, 4], [5, 6], [7]]
        identities = [(i % 7,) for i in range(count)]  # 7, 8 duplicate 0, 1
        excluded = torch.rand(count, count, generator=generator) < 0.2
        masks = info_nce_masks(
            clique_positive_mask(cliques, count), identities, excluded
        )

        result = info_nce_loss(units, masks, 0.3)

        expected = _reference_info_nce(units, masks, 0.3)
        assert result.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_an_anchor_without_negatives_contributes_zero_with_finite_gradient(
        self,
    ) -> None:
        # Anchor 0's only other unit is its positive: no negative at all.
        # Anchor 2 has one negative. Mean over the two anchors with
        # positives counts anchor 0 as 0
        torch.manual_seed(0)
        units = torch.randn(3, 4, requires_grad=True)
        positive = torch.zeros(3, 3, dtype=torch.bool)
        positive[0, 1] = positive[2, 1] = True
        negative = torch.zeros(3, 3, dtype=torch.bool)
        negative[2, 0] = True
        masks = ComparisonMasks(positive, negative)

        loss = info_nce_loss(units, masks, 0.07)
        loss.backward()

        assert units.grad is not None
        assert bool(torch.isfinite(units.grad).all())
        # Absolute tolerance too: at temperature 0.07 the loss can be ~0
        assert loss.item() == pytest.approx(
            _reference_info_nce(units.detach(), masks, 0.07).item(), rel=1e-5, abs=1e-6
        )

    def test_raises_when_no_anchor_has_a_positive(self) -> None:
        empty = torch.zeros(2, 2, dtype=torch.bool)
        masks = ComparisonMasks(empty, ~torch.eye(2, dtype=torch.bool))
        with pytest.raises(DegenerateBatchError):
            info_nce_loss(torch.randn(2, 4), masks, 0.07)

    def test_raises_on_a_unit_count_mismatch(self) -> None:
        masks = info_nce_masks(
            clique_positive_mask([[0, 1]], 3), [(1,), (2,), (3,)], None
        )
        with pytest.raises(ValueError, match="units"):
            info_nce_loss(torch.randn(2, 4), masks, 0.07)


class TestConstantLogitBaseline:
    def test_is_the_mean_log_of_one_plus_each_anchors_negatives(self) -> None:
        masks = info_nce_masks(
            clique_positive_mask([[0, 1], [2, 3]], 4), [(i,) for i in range(4)], None
        )
        # Every anchor: 1 positive, 2 negatives
        assert constant_logit_baseline(masks) == pytest.approx(math.log(3))

    def test_raises_when_no_anchor_has_a_negative(self) -> None:
        masks = info_nce_masks(clique_positive_mask([[0, 1]], 2), [(0,), (1,)], None)
        with pytest.raises(DegenerateBatchError, match="negative"):
            constant_logit_baseline(masks)


class TestCalculate:
    def test_raises_on_a_multi_card_shaped_input(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        with pytest.raises(ValueError):
            loss_fn.calculate(
                item_embeddings=_multis([[torch.randn(4), torch.randn(4)]]),
                identities=[(uuid4(),)],
                positive_cliques=[],
            )

    def test_raises_on_length_mismatch(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        with pytest.raises(ValueError):
            loss_fn.calculate(
                item_embeddings=_singles([torch.randn(4)]),
                identities=[(uuid4(),), (uuid4(),)],
                positive_cliques=[],
            )

    def test_raises_when_no_item_has_a_positive(self) -> None:
        loss_fn = SingleCardInfoNCELoss()

        with pytest.raises(ValueError):
            loss_fn.calculate(
                item_embeddings=_singles([torch.randn(4), torch.randn(4)]),
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
        identities = [(uuid4(),) for _ in range(4)]
        positive_cliques = [[0, 1], [2, 3]]

        result = loss_fn.calculate(_singles(torch.eye(4)), identities, positive_cliques)

        assert result.item() == pytest.approx(math.log(3), rel=1e-5)

    def test_excludes_a_duplicate_identity_from_the_negative_pool(self) -> None:
        # Item 2 is an exact-identity duplicate of item 0 (e.g. the same
        # card drawn from a different deck) but shares no positive group
        # with it. Anchor 0's only valid negative is then item 3, not
        # item 2.
        loss_fn = SingleCardInfoNCELoss(temperature=1.0)
        shared_uuid = uuid4()
        identities = [(shared_uuid,), (uuid4(),), (shared_uuid,), (uuid4(),)]
        positive_cliques = [[0, 1]]

        result = loss_fn.calculate(_singles(torch.eye(4)), identities, positive_cliques)

        # Anchor 0: positive=1, valid negatives={3} -> 2-way softmax.
        # Anchor 1: positive=0, valid negatives={2, 3} -> 3-way softmax.
        expected = (math.log(2) + math.log(3)) / 2
        assert result.item() == pytest.approx(expected, rel=1e-5)

    def test_accepts_a_composite_key_identity(self) -> None:
        # A composite loss (slice match) passes a pooled item's whole
        # sorted identity as one single-card key
        identities = [(uuid4(), uuid4()) for _ in range(4)]

        result = SingleCardInfoNCELoss().calculate(
            _singles(torch.randn(4, 3)), identities, [[0, 1], [2, 3]]
        )

        assert math.isfinite(result.item())


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
        loss = SingleCardInfoNCELoss().calculate(
            _singles([same] * 7), identities, cliques
        )
        baseline = SingleCardInfoNCELoss().constant_logit_loss(identities, cliques)
        assert loss.item() == pytest.approx(baseline)

    def test_a_duplicate_identity_is_not_counted_as_a_negative(self) -> None:
        identities = _single_identities(4)
        identities[3] = identities[0]  # item 3 is the same card as anchor 0
        cliques = [[0, 1], [2, 3]]
        same = torch.ones(4)
        loss = SingleCardInfoNCELoss().calculate(
            _singles([same] * 4), identities, cliques
        )
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
            _multis([[same, same] for _ in range(5)]), identities, cliques
        )
        baseline = CrossSliceCardInfoNCELoss().constant_logit_loss(identities, cliques)
        assert loss.item() == pytest.approx(baseline)

    def test_own_item_cards_are_neither_positive_nor_negative(self) -> None:
        identities = _multi_identities(3, 2)
        cliques = [[0, 1], [2]]
        # Card 0 (item 0) is aligned with item 1's cards and orthogonal to
        # its own sibling, which must not count as a negative
        a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        embeddings = _multis([[a, b], [a, a], [b, b]])
        loss = CrossSliceCardInfoNCELoss(temperature=1.0).calculate(
            embeddings, identities, cliques
        )
        assert torch.isfinite(loss)

    def test_raises_on_a_single_card_shaped_input(self) -> None:
        with pytest.raises(ValueError, match="multi-card"):
            CrossSliceCardInfoNCELoss().calculate(
                _singles([torch.randn(4), torch.randn(4)]),
                _multi_identities(2, 1),
                [[0, 1]],
            )

    def test_raises_when_a_mod_dropped_cards_from_an_item(self) -> None:
        # Item 0 kept its 3-card identity but holds only 2 cards
        embeddings = _multis([[torch.randn(4)] * 2, [torch.randn(4)] * 3])
        with pytest.raises(ValueError, match="one identity per card"):
            CrossSliceCardInfoNCELoss().calculate(
                embeddings, _multi_identities(2, 3), [[0, 1]]
            )
