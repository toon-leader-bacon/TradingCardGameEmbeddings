"""The slice-based contrastive styles beyond missing card: the shared
batch helpers, DeckSlicesPairConstructor (cross-slice card match and slice
match), card in contexts and odd one out."""

import logging
import math
import random
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.contrastive.contrastive_loss import DegenerateBatchError
from src.dojos.contrastive.contrastive_style import ContrastiveStyle
from src.dojos.contrastive.staple_subsampling import (
    DocumentFrequency,
    StapleSubsampling,
)
from src.dojos.contrastive.pair_constructor import (
    SliceSampler,
    consecutive_slices,
    contrastive_batch_from_deck_items,
)
from src.dojos.contrastive.styles.card_in_contexts import (
    AnchorCardLoss,
    CardInContextsPairConstructor,
    CardInContextsStyle,
)
from src.dojos.contrastive.styles.cross_slice_card import (
    CrossSliceCardInfoNCELoss,
    CrossSliceCardStyle,
)
from src.dojos.contrastive.styles.deck_slices import DeckSlicesPairConstructor
from src.dojos.contrastive.styles.odd_one_out import (
    OddOneOutLoss,
    OddOneOutPairConstructor,
    OddOneOutStyle,
)
from src.dojos.contrastive.styles.single_card import SingleCardInfoNCELoss
from src.dojos.contrastive.styles.slice_match import PooledItemsLoss, SliceMatchStyle
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(binder: CardBinder, name: str) -> GenericCard:
    card = GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    binder.create(card)
    return card


def _deck(card_uuids: list[UUID]) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="deck",
        card_nocab_uuids=card_uuids,
        provenance=None,
    )


def _binder_and_decks(
    deck_count: int, cards_per_deck: int
) -> tuple[CardBinder, list[GenericDeck]]:
    binder = CardBinder()
    decks = [
        _deck([_card(binder, f"d{d}c{c}").nocab_uuid for c in range(cards_per_deck)])
        for d in range(deck_count)
    ]
    return binder, decks


def _random_embeddings(identities: list[tuple[UUID, ...]]) -> list[list[torch.Tensor]]:
    return [[torch.randn(8) for _ in identity] for identity in identities]


def _constant_embeddings(
    identities: list[tuple[UUID, ...]],
) -> list[list[torch.Tensor]]:
    same = torch.tensor([1.0, 2.0, 3.0])
    return [[same] * len(identity) for identity in identities]


class TestSharedHelpers:
    def test_consecutive_slices_cut_in_order(self) -> None:
        a, b, c, d = (uuid4() for _ in range(4))
        assert consecutive_slices([a, b, c, d], 2) == [(a, b), (c, d)]

    @pytest.mark.parametrize("size", [0, 3])
    def test_consecutive_slices_need_an_exact_multiple(self, size: int) -> None:
        with pytest.raises(ValueError):
            consecutive_slices([uuid4() for _ in range(4)], size)

    def test_batch_from_deck_items_keeps_item_and_clique_order(self) -> None:
        binder, (deck,) = _binder_and_decks(1, 3)
        a1, a2, a3 = deck.card_nocab_uuids

        batch = contrastive_batch_from_deck_items([[(a1, a2), (a3,)]], binder)

        assert batch.identities == [(a1, a2), (a3,)]
        assert batch.positive_cliques == [[0, 1]]
        assert [card.nocab_uuid for card in batch.inputs[0]] == [a1, a2]  # type: ignore[union-attr]

    def test_batch_from_no_deck_items_is_empty(self) -> None:
        batch = contrastive_batch_from_deck_items([], CardBinder())
        assert batch.inputs == [] and batch.positive_cliques == []

    def test_held_out_draw_excludes_every_copy_of_the_pick(self) -> None:
        binder, (deck,) = _binder_and_decks(1, 0)
        picked = uuid4()
        others = [uuid4() for _ in range(4)]
        sampler = SliceSampler(random.Random(0))

        # 4 others always leave 2 to draw, whatever is picked
        for _ in range(10):
            draw = sampler.held_out_draw(deck, [picked] * 3 + others, 2)
            assert draw is not None
            assert draw.held_out not in draw.rest


class TestDeckSlicesPairConstructor:
    @pytest.mark.parametrize(("size", "count"), [(0, 2), (2, 1)])
    def test_rejects_bad_sizes(self, size: int, count: int) -> None:
        with pytest.raises(ValueError):
            DeckSlicesPairConstructor(size, count)

    def test_builds_disjoint_same_deck_slices(self) -> None:
        binder, decks = _binder_and_decks(2, 10)

        batch = DeckSlicesPairConstructor(4, 2, rng_seed=0).build(decks, binder)

        assert batch.positive_cliques == [[0, 1], [2, 3]]
        assert [len(item) for item in batch.inputs] == [4, 4, 4, 4]  # type: ignore[arg-type]
        for (first, second), deck in zip(batch.positive_cliques, decks):
            assert not set(batch.identities[first]) & set(batch.identities[second])
            assert set(batch.identities[first]) <= set(deck.card_nocab_uuids)

    def test_skips_a_deck_too_small_for_every_slice(self) -> None:
        binder, (small,) = _binder_and_decks(1, 7)
        big = _deck([_card(binder, f"b{i}").nocab_uuid for i in range(8)])

        batch = DeckSlicesPairConstructor(4, 2, rng_seed=0).build([small, big], binder)

        assert batch.positive_cliques == [[0, 1]]

    def test_a_deck_costs_every_slice(self) -> None:
        assert DeckSlicesPairConstructor(4, 3).cards_per_deck == 12


class TestCrossSliceAndSliceMatchStyles:
    def test_cross_slice_style_pairs_its_parts(self) -> None:
        style = CrossSliceCardStyle()
        assert isinstance(style.pair_constructor(0, None), DeckSlicesPairConstructor)
        assert style.pair_constructor(0, None).cards_per_deck == 8
        assert isinstance(style.contrastive_loss(), CrossSliceCardInfoNCELoss)

    def test_slice_match_style_pairs_its_parts(self) -> None:
        style = SliceMatchStyle()
        assert style.pair_constructor(0, None).cards_per_deck == 8
        assert isinstance(style.contrastive_loss(), PooledItemsLoss)

    @pytest.mark.parametrize("style", [CrossSliceCardStyle(), SliceMatchStyle()])
    def test_a_built_batch_scores_at_its_baseline_when_constant(
        self, style: ContrastiveStyle
    ) -> None:
        binder, decks = _binder_and_decks(4, 10)
        batch = style.pair_constructor(0, None).build(decks, binder)
        loss_fn = style.contrastive_loss()

        loss = loss_fn.calculate(
            _constant_embeddings(batch.identities),
            batch.identities,
            batch.positive_cliques,
        )

        assert loss.item() == pytest.approx(
            loss_fn.constant_logit_loss(batch.identities, batch.positive_cliques),
            rel=1e-5,
        )


class TestPooledItemsLoss:
    def test_baseline_counts_slices_like_single_cards(self) -> None:
        # 3 decks x 2 slices, no duplicates: ln(6 - 2 + 1) = ln 5
        identities = [(uuid4(), uuid4()) for _ in range(6)]

        baseline = PooledItemsLoss(SingleCardInfoNCELoss()).constant_logit_loss(
            identities, [[0, 1], [2, 3], [4, 5]]
        )

        assert baseline == pytest.approx(math.log(5))

    def test_the_same_cards_in_another_order_are_not_negatives(self) -> None:
        # Slices 1 and 3 hold the same two cards in opposite orders
        a, b = uuid4(), uuid4()
        identities = [(uuid4(), uuid4()), (a, b), (uuid4(), uuid4()), (b, a)]
        loss_fn = PooledItemsLoss(SingleCardInfoNCELoss())

        baseline = loss_fn.constant_logit_loss(identities, [[0, 1], [2, 3]])

        # Items 1 and 3 each lose the other as a negative: 2 x ln 3 + 2 x ln 2
        assert baseline == pytest.approx((2 * math.log(3) + 2 * math.log(2)) / 4)

    def test_raises_on_a_single_card_shaped_input(self) -> None:
        with pytest.raises(ValueError, match="multi-card"):
            PooledItemsLoss(SingleCardInfoNCELoss()).calculate(
                [torch.randn(4), torch.randn(4)], [(uuid4(),), (uuid4(),)], [[0, 1]]
            )


class TestCardInContexts:
    @pytest.mark.parametrize(("size", "count"), [(0, 2), (3, 1)])
    def test_rejects_bad_sizes(self, size: int, count: int) -> None:
        with pytest.raises(ValueError):
            CardInContextsPairConstructor(size, count)

    def test_each_item_is_the_anchor_then_a_disjoint_context(self) -> None:
        binder, decks = _binder_and_decks(2, 10)

        batch = CardInContextsPairConstructor(3, 2, rng_seed=0).build(decks, binder)

        assert batch.positive_cliques == [[0, 1], [2, 3]]
        for first, second in batch.positive_cliques:
            anchor = batch.identities[first][0]
            assert batch.identities[second][0] == anchor
            first_context = set(batch.identities[first][1:])
            second_context = set(batch.identities[second][1:])
            assert anchor not in first_context | second_context
            assert not first_context & second_context

    def test_a_deck_costs_the_anchor_per_context_plus_the_contexts(self) -> None:
        assert CardInContextsStyle().pair_constructor(0, None).cards_per_deck == 16

    def test_anchor_loss_scores_only_the_anchor(self) -> None:
        # Anchors are matched per deck; every context card is noise that
        # must not change the loss
        identities = [(uuid4(), uuid4()) for _ in range(4)]
        x, y = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        cliques = [[0, 1], [2, 3]]
        loss_fn = AnchorCardLoss(SingleCardInfoNCELoss(temperature=0.1))

        quiet = loss_fn.calculate([[x, x], [x, x], [y, y], [y, y]], identities, cliques)
        noisy = loss_fn.calculate(
            [[x, y], [x, -x], [y, x], [y, -y]], identities, cliques
        )

        assert quiet.item() == pytest.approx(noisy.item())

    def test_two_decks_on_the_same_anchor_are_not_negatives(self) -> None:
        # Decks A and B are both anchored on card s, deck C on card t.
        # Whole-item identities all differ; the anchors' don't
        s, t = uuid4(), uuid4()
        identities = [(anchor, uuid4()) for anchor in (s, s, s, s, t, t)]

        baseline = AnchorCardLoss(SingleCardInfoNCELoss()).constant_logit_loss(
            identities, [[0, 1], [2, 3], [4, 5]]
        )

        # A's and B's 4 anchors lose each other as negatives (2 left each:
        # C's), C's 2 anchors keep all 4 of A's and B's: mean of 4 x ln 3
        # and 2 x ln 5 (unmapped identities would give ln 5 for all six)
        assert baseline == pytest.approx((4 * math.log(3) + 2 * math.log(5)) / 6)

    def test_style_scores_at_its_baseline_when_constant(self) -> None:
        binder, decks = _binder_and_decks(4, 20)
        style = CardInContextsStyle()
        batch = style.pair_constructor(0, None).build(decks, binder)
        loss_fn = style.contrastive_loss()

        loss = loss_fn.calculate(
            _constant_embeddings(batch.identities),
            batch.identities,
            batch.positive_cliques,
        )

        assert loss.item() == pytest.approx(
            loss_fn.constant_logit_loss(batch.identities, batch.positive_cliques),
            rel=1e-5,
        )


class TestOddOneOutPairConstructor:
    def test_rejects_a_slice_under_three(self) -> None:
        with pytest.raises(ValueError):
            OddOneOutPairConstructor(2)

    def test_each_item_is_its_own_deck_plus_a_foreign_intruder_last(self) -> None:
        binder, decks = _binder_and_decks(3, 10)

        batch = OddOneOutPairConstructor(5, rng_seed=0).build(decks, binder)

        assert batch.positive_cliques == [[0], [1], [2]]
        for identity, deck in zip(batch.identities, decks):
            *host, intruder = identity
            assert len(host) == 4
            assert set(host) <= set(deck.card_nocab_uuids)
            assert intruder not in deck.card_nocab_uuids

    def test_the_intruder_is_never_a_card_the_host_shares_with_the_donor(
        self,
    ) -> None:
        binder = CardBinder()
        shared = [_card(binder, f"s{i}").nocab_uuid for i in range(5)]
        host = _deck(shared + [_card(binder, f"h{i}").nocab_uuid for i in range(5)])
        donor = _deck(shared + [_card(binder, f"d{i}").nocab_uuid for i in range(5)])

        for seed in range(20):
            batch = OddOneOutPairConstructor(4, rng_seed=seed).build(
                [host, donor], binder
            )
            for identity, deck in zip(batch.identities, (host, donor)):
                assert identity[-1] not in deck.card_nocab_uuids

    def test_a_host_card_thinned_out_by_staple_subsampling_is_still_no_intruder(
        self,
    ) -> None:
        # Each staple occurrence is kept with p = sqrt(0.09 / 1.0) = 0.3, so
        # often the host's candidates lose its staple while the donor keeps
        # one. Every donor card is in the host deck, so the host has no
        # valid intruder and must be skipped; checking intruders against
        # the host's thinned candidates (not every card it holds) would
        # hand it the staple instead
        binder = CardBinder()
        staple = _card(binder, "staple").nocab_uuid
        host_cards = [_card(binder, f"h{i}").nocab_uuid for i in range(6)]
        host = _deck([staple] + host_cards)
        donor = _deck([staple] * 3 + host_cards[:3])
        frequency = DocumentFrequency(
            card_binder_version="v", deck_count=1, shares={staple: 1.0}
        )
        thinning = StapleSubsampling(threshold=0.09, frequency=frequency)

        for seed in range(30):
            batch = OddOneOutPairConstructor(
                4, rng_seed=seed, staple_subsampling=thinning
            ).build([host, donor], binder)
            assert all(identity[-1] != staple for identity in batch.identities)

    def test_same_seed_same_batch(self) -> None:
        binder, decks = _binder_and_decks(4, 10)

        first = OddOneOutPairConstructor(5, rng_seed=3).build(decks, binder)
        second = OddOneOutPairConstructor(5, rng_seed=3).build(decks, binder)

        assert first.identities == second.identities

    def test_one_surviving_deck_has_no_donor(self) -> None:
        binder, decks = _binder_and_decks(1, 10)
        batch = OddOneOutPairConstructor(5, rng_seed=0).build(decks, binder)
        assert batch.inputs == []

    def test_skips_a_host_whose_donor_holds_only_its_cards(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder, (host,) = _binder_and_decks(1, 6)
        twin = _deck(list(host.card_nocab_uuids))  # the same cards

        with caplog.at_level(logging.DEBUG):
            batch = OddOneOutPairConstructor(4, rng_seed=0).build([host, twin], binder)

        assert batch.inputs == []
        assert "holds no card it lacks" in caplog.text


class TestOddOneOutLoss:
    def test_baseline_is_ln_of_the_item_size(self) -> None:
        identities = [tuple(uuid4() for _ in range(8)) for _ in range(3)]

        baseline = OddOneOutLoss().constant_logit_loss(identities, [[0], [1], [2]])

        assert baseline == pytest.approx(math.log(8))

    def test_constant_embeddings_score_exactly_the_baseline(self) -> None:
        identities = [tuple(uuid4() for _ in range(5)) for _ in range(2)]
        loss_fn = OddOneOutLoss()

        loss = loss_fn.calculate(
            _constant_embeddings(identities), identities, [[0], [1]]
        )

        assert loss.item() == pytest.approx(math.log(5), rel=1e-5)

    def test_a_clear_intruder_scores_far_below_the_baseline(self) -> None:
        identities = [tuple(uuid4() for _ in range(4))]
        x, y = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])

        loss = OddOneOutLoss(temperature=0.1).calculate(
            [[x, x, x, y]], identities, [[0]]
        )

        assert loss.item() < 0.01

    def test_gradients_reach_every_card(self) -> None:
        identities = [tuple(uuid4() for _ in range(4))]
        cards = [torch.randn(4, requires_grad=True) for _ in range(4)]

        OddOneOutLoss().calculate([cards], identities, [[0]]).backward()

        assert all(card.grad is not None for card in cards)

    def test_an_empty_batch_is_degenerate(self) -> None:
        with pytest.raises(DegenerateBatchError):
            OddOneOutLoss().constant_logit_loss([], [])

    @pytest.mark.parametrize(
        ("identity_sizes", "cliques"),
        [([4, 4], [[0, 1]]), ([4, 4], [[0]]), ([2], [[0]])],
        ids=["not-singleton", "uncovered", "too-small"],
    )
    def test_rejects_a_malformed_layout(
        self, identity_sizes: list[int], cliques: list[list[int]]
    ) -> None:
        identities = [tuple(uuid4() for _ in range(size)) for size in identity_sizes]
        with pytest.raises(ValueError):
            OddOneOutLoss().constant_logit_loss(identities, cliques)

    def test_rejects_an_intruder_that_also_sits_in_the_slice(self) -> None:
        intruder = uuid4()
        identities = [(intruder, uuid4(), intruder)]
        with pytest.raises(ValueError, match="intruder"):
            OddOneOutLoss().constant_logit_loss(identities, [[0]])

    def test_style_end_to_end(self) -> None:
        binder, decks = _binder_and_decks(4, 10)
        style = OddOneOutStyle()
        batch = style.pair_constructor(0, None).build(decks, binder)

        loss = style.contrastive_loss().calculate(
            _random_embeddings(batch.identities),
            batch.identities,
            batch.positive_cliques,
        )

        assert torch.isfinite(loss)
        assert style.pair_constructor(0, None).cards_per_deck == 8
