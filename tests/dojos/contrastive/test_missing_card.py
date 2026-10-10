"""The missing-card contrastive style: SliceSampler, MissingCardPairConstructor,
MissingCardInfoNCELoss and the ContrastiveStyle pairs."""

import logging
import math
import random
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.contrastive.pair_constructor import HeldOutDraw, SliceSampler
from src.dojos.contrastive.styles.missing_card import (
    MissingCardInfoNCELoss,
    MissingCardPairConstructor,
    MissingCardStyle,
)
from src.dojos.contrastive.styles.single_card import (
    SingleCardInfoNCELoss,
    SingleCardPairConstructor,
    SingleCardStyle,
)
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


def _deck(card_uuids: list[UUID], name: str = "deck") -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        card_nocab_uuids=card_uuids,
        provenance=None,
    )


def _binder_and_decks(
    deck_count: int, cards_per_deck: int
) -> tuple[CardBinder, list[GenericDeck]]:
    binder = CardBinder()
    decks = [
        _deck(
            [_card(binder, f"d{d}c{c}").nocab_uuid for c in range(cards_per_deck)],
            name=f"deck{d}",
        )
        for d in range(deck_count)
    ]
    return binder, decks


class TestSliceSampler:
    def test_candidates_are_the_known_cards(self) -> None:
        binder, (deck,) = _binder_and_decks(1, 5)
        unknown = uuid4()
        deck = _deck(deck.card_nocab_uuids + [unknown])
        sampler = SliceSampler(random.Random(0))

        assert sampler.candidate_uuids(deck, binder, minimum=5) == (
            deck.card_nocab_uuids[:5]
        )

    def test_too_few_known_cards_is_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder, (deck,) = _binder_and_decks(1, 3)
        sampler = SliceSampler(random.Random(0))

        with caplog.at_level(logging.DEBUG):
            assert sampler.candidate_uuids(deck, binder, minimum=4) is None

        assert [record.levelno for record in caplog.records] == [logging.WARNING]

    def test_draw_leaves_out_every_copy_of_an_excluded_card(self) -> None:
        excluded, kept = uuid4(), [uuid4() for _ in range(3)]
        sampler = SliceSampler(random.Random(0))

        drawn = sampler.draw([excluded, *kept, excluded], 3, frozenset({excluded}))

        assert drawn is not None and sorted(drawn) == sorted(kept)

    def test_draw_is_none_when_the_exclusion_leaves_too_few(self) -> None:
        excluded = uuid4()
        sampler = SliceSampler(random.Random(0))

        assert (
            sampler.draw([excluded, excluded, uuid4()], 2, frozenset({excluded}))
            is None
        )

    def test_draw_raises_on_a_non_positive_size(self) -> None:
        with pytest.raises(ValueError):
            SliceSampler(random.Random(0)).draw([uuid4()], 0)

    def test_single_card_draws_match_plain_rng_sample(self) -> None:
        # Moving SingleCardPairConstructor onto SliceSampler kept its draws
        binder, (deck,) = _binder_and_decks(1, 10)

        batch = SingleCardPairConstructor(items_per_deck=3, rng_seed=7).build(
            [deck], binder
        )

        expected = random.Random(7).sample(deck.card_nocab_uuids, 3)
        assert [identity[0] for identity in batch.identities] == expected


class TestHeldOutDraw:
    def test_rejects_a_held_out_card_among_the_rest(self) -> None:
        card = uuid4()
        with pytest.raises(ValueError):
            HeldOutDraw(card, (card, uuid4()))


class TestMissingCardPairConstructor:
    def test_rejects_a_slice_smaller_than_two(self) -> None:
        with pytest.raises(ValueError):
            MissingCardPairConstructor(slice_size=1)

    def test_a_deck_costs_its_slice_plus_the_missing_card(self) -> None:
        assert MissingCardPairConstructor(slice_size=8).cards_per_deck == 9

    def test_builds_a_context_then_card_pair_per_deck(self) -> None:
        binder, decks = _binder_and_decks(3, 10)
        constructor = MissingCardPairConstructor(slice_size=4, rng_seed=0)

        batch = constructor.build(decks, binder)

        assert batch.positive_cliques == [[0, 1], [2, 3], [4, 5]]
        assert [len(item) for item in batch.inputs] == [4, 1, 4, 1, 4, 1]
        for (context_index, card_index), deck in zip(batch.positive_cliques, decks):
            context = set(batch.identities[context_index])
            (missing,) = batch.identities[card_index]
            assert context <= set(deck.card_nocab_uuids)
            assert missing in deck.card_nocab_uuids
            assert missing not in context

    def test_no_copy_of_the_missing_card_stays_in_the_slice(self) -> None:
        binder = CardBinder()
        four_of = _card(binder, "four_of").nocab_uuid
        others = [_card(binder, f"c{i}").nocab_uuid for i in range(3)]
        deck = _deck([four_of] * 4 + others)
        constructor = MissingCardPairConstructor(slice_size=3, rng_seed=0)

        # 4 + 3 cards at slice 3 always leaves enough, whatever is picked
        for _ in range(20):
            batch = constructor.build([deck], binder)
            assert batch.positive_cliques == [[0, 1]]
            (missing,) = batch.identities[1]
            assert missing not in batch.identities[0]

    def test_skips_at_debug_when_the_picks_copies_leave_too_few(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [X X X a b] at slice 3: picking X leaves only [a b]
        binder = CardBinder()
        three_of = _card(binder, "three_of").nocab_uuid
        deck = _deck([three_of] * 3 + [_card(binder, n).nocab_uuid for n in "ab"])
        constructor = MissingCardPairConstructor(slice_size=3, rng_seed=0)

        with caplog.at_level(logging.DEBUG):
            batches = [constructor.build([deck], binder) for _ in range(20)]

        skipped = [batch for batch in batches if not batch.positive_cliques]
        assert skipped, "X is picked 3 times in 5; 20 builds should hit it"
        assert all(
            batch.identities[1] != (three_of,)
            for batch in batches
            if batch.positive_cliques
        )
        assert {record.levelno for record in caplog.records} == {logging.DEBUG}

    def test_skips_a_deck_too_small_for_a_pick_and_a_slice(self) -> None:
        binder, (small, big) = _binder_and_decks(2, 0)
        small = _deck([_card(binder, f"s{i}").nocab_uuid for i in range(4)])
        big = _deck([_card(binder, f"b{i}").nocab_uuid for i in range(9)])

        batch = MissingCardPairConstructor(slice_size=4, rng_seed=0).build(
            [small, big], binder
        )

        assert batch.positive_cliques == [[0, 1]]
        assert set(batch.identities[0]) <= set(big.card_nocab_uuids)

    def test_same_seed_same_batch(self) -> None:
        binder, decks = _binder_and_decks(3, 10)

        first = MissingCardPairConstructor(4, rng_seed=3).build(decks, binder)
        second = MissingCardPairConstructor(4, rng_seed=3).build(decks, binder)

        assert first.identities == second.identities


def _pair_identities(missing_cards: list[UUID]) -> list[tuple[UUID, ...]]:
    """[context, card] identities per pair: a 2-card context, then the card."""
    result: list[tuple[UUID, ...]] = []
    for missing in missing_cards:
        result.append((uuid4(), uuid4()))
        result.append((missing,))
    return result


def _cliques(pair_count: int) -> list[list[int]]:
    return [[2 * i, 2 * i + 1] for i in range(pair_count)]


class TestMissingCardInfoNCELoss:
    def test_baseline_is_ln_of_the_pair_count(self) -> None:
        identities = _pair_identities([uuid4() for _ in range(3)])

        baseline = MissingCardInfoNCELoss().constant_logit_loss(identities, _cliques(3))

        assert baseline == pytest.approx(math.log(3))

    def test_two_decks_missing_the_same_card_are_not_each_others_negatives(
        self,
    ) -> None:
        shared = uuid4()
        identities = _pair_identities([shared, shared, uuid4()])

        baseline = MissingCardInfoNCELoss().constant_logit_loss(identities, _cliques(3))

        # Each of the twin decks' 4 anchors keeps 1 negative, the third
        # deck's 2 anchors keep 2: mean of 4 x ln 2 and 2 x ln 3
        assert baseline == pytest.approx((4 * math.log(2) + 2 * math.log(3)) / 6)

    @pytest.mark.parametrize("twins", [False, True])
    def test_constant_embeddings_score_exactly_the_baseline(self, twins: bool) -> None:
        # Equal embeddings make every logit equal: calculate() must then
        # agree with constant_logit_loss(), twin exclusions included
        shared = uuid4()
        missing = [shared, shared, uuid4()] if twins else [uuid4() for _ in range(3)]
        identities = _pair_identities(missing)
        same = torch.tensor([1.0, 2.0, 3.0])
        embeddings = [[same] * len(identity) for identity in identities]
        loss_fn = MissingCardInfoNCELoss()

        loss = loss_fn.calculate(embeddings, identities, _cliques(3))

        assert loss.item() == pytest.approx(
            loss_fn.constant_logit_loss(identities, _cliques(3)), rel=1e-5
        )

    def test_raises_on_an_item_in_no_clique(self) -> None:
        identities = _pair_identities([uuid4()]) + [(uuid4(),)]
        with pytest.raises(ValueError, match="every item"):
            MissingCardInfoNCELoss().constant_logit_loss(identities, _cliques(1))

    def test_matched_contexts_and_cards_score_below_the_baseline(self) -> None:
        identities = _pair_identities([uuid4() for _ in range(3)])
        axes = torch.eye(3)
        # Each context's cards average to its own card's direction
        embeddings = [
            item
            for i in range(3)
            for item in ([axes[i] + 0.1, axes[i] - 0.1], [axes[i]])
        ]
        loss_fn = MissingCardInfoNCELoss(temperature=0.1)

        loss = loss_fn.calculate(embeddings, identities, _cliques(3))

        assert loss.item() < loss_fn.constant_logit_loss(identities, _cliques(3))

    def test_contexts_only_meet_cards_and_cards_only_meet_contexts(self) -> None:
        # Both contexts point along x; card 0 along x, card 1 along y.
        # Were the two contexts (similarity 1) ever compared, context 0
        # would gain a strong negative and the loss would differ
        identities = _pair_identities([uuid4() for _ in range(2)])
        x, y = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        embeddings = [[x, x], [x], [x, x], [y]]

        loss = MissingCardInfoNCELoss(temperature=0.1).calculate(
            embeddings, identities, _cliques(2)
        )

        # Logits are similarity / 0.1. Context 0: positive 10 vs negative
        # 0. Context 1: positive 0 vs negative 10. Card 0: positive 10 vs
        # negative 10. Card 1: positive 0 vs negative 0.
        expected = (
            math.log(1 + math.exp(-10))
            + math.log(1 + math.exp(10))
            + math.log(2)
            + math.log(2)
        ) / 4
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_gradients_reach_every_embedding(self) -> None:
        identities = _pair_identities([uuid4() for _ in range(2)])
        embeddings = [
            [torch.randn(4, requires_grad=True) for _ in range(len(identity))]
            for identity in identities
        ]

        MissingCardInfoNCELoss().calculate(
            embeddings, identities, _cliques(2)
        ).backward()

        assert all(card.grad is not None for item in embeddings for card in item)

    def test_raises_on_a_single_card_shaped_input(self) -> None:
        with pytest.raises(ValueError, match="multi-card"):
            MissingCardInfoNCELoss().calculate(
                [torch.randn(4), torch.randn(4)], [(uuid4(),), (uuid4(),)], [[0, 1]]
            )

    @pytest.mark.parametrize(
        "cliques", [[], [[0, 1, 2]], [[1, 0]]], ids=["none", "triple", "reversed"]
    )
    def test_raises_on_a_clique_that_is_not_a_context_card_pair(
        self, cliques: list[list[int]]
    ) -> None:
        identities = [(uuid4(), uuid4()), (uuid4(),), (uuid4(),)]
        with pytest.raises(ValueError):
            MissingCardInfoNCELoss().constant_logit_loss(identities, cliques)


class TestEndToEnd:
    def test_a_built_batch_scores_with_its_own_loss(self) -> None:
        binder, decks = _binder_and_decks(4, 10)
        style = MissingCardStyle(slice_size=4)
        batch = style.pair_constructor(0, None).build(decks, binder)
        embeddings = [[torch.randn(8) for _ in item] for item in batch.inputs]
        loss_fn = style.contrastive_loss()

        loss = loss_fn.calculate(embeddings, batch.identities, batch.positive_cliques)

        assert torch.isfinite(loss)
        assert loss_fn.constant_logit_loss(
            batch.identities, batch.positive_cliques
        ) == pytest.approx(math.log(4))


class TestStyles:
    def test_single_card_style_pairs_single_card_parts(self) -> None:
        style = SingleCardStyle(items_per_deck=3)

        assert isinstance(style.pair_constructor(0, None), SingleCardPairConstructor)
        assert style.pair_constructor(0, None).cards_per_deck == 3
        assert isinstance(style.contrastive_loss(), SingleCardInfoNCELoss)

    def test_missing_card_style_pairs_missing_card_parts(self) -> None:
        style = MissingCardStyle(slice_size=8)

        assert isinstance(style.pair_constructor(0, None), MissingCardPairConstructor)
        assert style.pair_constructor(0, None).cards_per_deck == 9
        assert isinstance(style.contrastive_loss(), MissingCardInfoNCELoss)
