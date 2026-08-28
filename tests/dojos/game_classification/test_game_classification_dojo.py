import math
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import CardCount, Split
from src.dojos.game_classification.game_classification_dojo import GameClassificationDojo
from src.dojos.losses.cross_entropy_loss import CrossEntropyLoss
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str, source_game: GameId) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=source_game,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder(num_mtg: int, num_pokemon: int) -> CardBinder:
    binder = CardBinder()
    for i in range(num_mtg):
        binder.add(_card(f"mtg-{i}", GameId.MTG))
    for i in range(num_pokemon):
        binder.add(_card(f"pokemon-{i}", GameId.POKEMON))
    return binder


def _dojo(
    train_ratio: float = 0.5, validate_ratio: float = 0.2, rng_seed: int = 0
) -> GameClassificationDojo:
    return GameClassificationDojo(
        train_ratio=train_ratio,
        validate_ratio=validate_ratio,
        loss=CrossEntropyLoss(label_smoothing=0.0),
        rng_seed=rng_seed,
    )


class TestInit:
    @pytest.mark.parametrize("train_ratio", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_train_ratio_raises(self, train_ratio: float) -> None:
        with pytest.raises(ValueError):
            GameClassificationDojo(
                train_ratio=train_ratio,
                validate_ratio=0.1,
                loss=CrossEntropyLoss(0.0),
                rng_seed=0,
            )

    @pytest.mark.parametrize("validate_ratio", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_validate_ratio_raises(self, validate_ratio: float) -> None:
        with pytest.raises(ValueError):
            GameClassificationDojo(
                train_ratio=0.5,
                validate_ratio=validate_ratio,
                loss=CrossEntropyLoss(0.0),
                rng_seed=0,
            )

    def test_ratios_summing_to_one_or_more_raises(self) -> None:
        with pytest.raises(ValueError):
            GameClassificationDojo(
                train_ratio=0.5, validate_ratio=0.5, loss=CrossEntropyLoss(0.0), rng_seed=0
            )


class TestCardCount:
    def test_is_single_card(self) -> None:
        assert _dojo().card_count() == CardCount(minimum=1, maximum=1)


class TestBeforePrepareSplits:
    def test_split_size_raises(self) -> None:
        with pytest.raises(RuntimeError):
            _dojo().split_size(Split.TRAIN)

    def test_next_batch_raises(self) -> None:
        with pytest.raises(RuntimeError):
            _dojo().next_batch(Split.TRAIN, batch_size=1)


class TestPrepareSplits:
    def test_splits_partition_every_card_without_gaps(self) -> None:
        binder = _binder(num_mtg=15, num_pokemon=15)
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)

        dojo.prepare_splits(binder, held_out_cards=set())

        total = (
            dojo.split_size(Split.TRAIN)
            + dojo.split_size(Split.TEST)
            + dojo.split_size(Split.VALIDATE)
        )
        assert total == 30

    def test_held_out_cards_never_appear_in_train_draws(self) -> None:
        binder = _binder(num_mtg=10, num_pokemon=10)
        held_out_uuid = next(iter(binder.all_uuids()))
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)

        dojo.prepare_splits(binder, held_out_cards={held_out_uuid})

        # Draw far more than the train pool size, with replacement — if the
        # held-out uuid were ever eligible it would show up with high
        # probability across this many draws.
        for _ in range(20):
            batch = dojo.next_batch(Split.TRAIN, batch_size=dojo.split_size(Split.TRAIN) or 1)
            drawn_uuids = {cs.cards[0].nocab_uuid for cs in batch.card_sets}
            assert held_out_uuid not in drawn_uuids


class TestNextBatchTrain:
    def test_returns_requested_batch_size(self) -> None:
        binder = _binder(num_mtg=10, num_pokemon=10)
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())

        batch = dojo.next_batch(Split.TRAIN, batch_size=5)

        assert len(batch.card_sets) == 5
        assert len(batch.labels) == 5

    def test_each_card_set_holds_exactly_one_card(self) -> None:
        binder = _binder(num_mtg=10, num_pokemon=10)
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())

        batch = dojo.next_batch(Split.TRAIN, batch_size=3)

        for card_set in batch.card_sets:
            assert len(card_set.cards) == 1

    def test_labels_match_each_card_own_source_game(self) -> None:
        binder = _binder(num_mtg=10, num_pokemon=10)
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())

        batch = dojo.next_batch(Split.TRAIN, batch_size=5)

        for card_set, label in zip(batch.card_sets, batch.labels):
            assert card_set.cards[0].source_game == label


@pytest.mark.parametrize("split", [Split.TEST, Split.VALIDATE])
class TestNextBatchExhaustivePasses:
    def test_one_full_pass_covers_every_card_exactly_once(self, split: Split) -> None:
        binder = _binder(num_mtg=10, num_pokemon=10)
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())
        pool_size = dojo.split_size(split)
        assert pool_size > 0
        batch_size = 3

        seen_uuids: list = []
        for _ in range(math.ceil(pool_size / batch_size)):
            batch = dojo.next_batch(split, batch_size)
            seen_uuids.extend(cs.cards[0].nocab_uuid for cs in batch.card_sets)

        assert len(seen_uuids) == pool_size
        assert len(set(seen_uuids)) == pool_size

    def test_final_batch_of_a_pass_is_short_not_padded(self, split: Split) -> None:
        binder = _binder(num_mtg=10, num_pokemon=10)
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())
        pool_size = dojo.split_size(split)
        assert pool_size >= 2
        batch_size = pool_size - 1

        dojo.next_batch(split, batch_size)  # consumes pool_size - 1
        final_batch = dojo.next_batch(split, batch_size)  # only 1 left

        assert len(final_batch.card_sets) == 1

    def test_wraps_to_a_fresh_pass_after_full_coverage(self, split: Split) -> None:
        binder = _binder(num_mtg=10, num_pokemon=10)
        dojo = _dojo(train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())
        pool_size = dojo.split_size(split)
        batch_size = pool_size

        dojo.next_batch(split, batch_size)  # drains the whole pool
        next_pass_first_batch = dojo.next_batch(split, batch_size)

        assert len(next_pass_first_batch.card_sets) == pool_size


class TestBuildDecoderHead:
    def test_returns_linear_layer_sized_for_every_game(self) -> None:
        dojo = _dojo()

        head = dojo.build_decoder_head(embedding_dim=16)

        assert isinstance(head, torch.nn.Linear)
        assert head.in_features == 16
        assert head.out_features == len(GameId)


class TestGameIdToClassIndex:
    def test_every_game_id_maps_to_a_distinct_index(self) -> None:
        dojo = _dojo()

        indices = {dojo._game_id_to_class_index(game_id) for game_id in GameId}

        assert indices == set(range(len(GameId)))

    def test_mapping_is_stable_across_calls(self) -> None:
        dojo = _dojo()

        first = dojo._game_id_to_class_index(GameId.MTG)
        second = dojo._game_id_to_class_index(GameId.MTG)

        assert first == second


class TestComputeLoss:
    def test_before_build_decoder_head_raises(self) -> None:
        dojo = _dojo()

        with pytest.raises(ValueError):
            dojo.compute_loss([[torch.randn(4)]], [GameId.MTG])

    def test_mismatched_lengths_raises(self) -> None:
        dojo = _dojo()
        dojo.build_decoder_head(embedding_dim=4)

        with pytest.raises(ValueError):
            dojo.compute_loss([[torch.randn(4)]], [GameId.MTG, GameId.POKEMON])

    def test_wrong_cards_per_example_raises(self) -> None:
        dojo = _dojo()
        dojo.build_decoder_head(embedding_dim=4)

        with pytest.raises(ValueError):
            dojo.compute_loss([[torch.randn(4), torch.randn(4)]], [GameId.MTG])

    def test_returns_scalar_and_backpropagates(self) -> None:
        dojo = _dojo()
        head = dojo.build_decoder_head(embedding_dim=4)
        embeddings = [[torch.randn(4, requires_grad=True)] for _ in range(3)]
        labels = [GameId.MTG, GameId.POKEMON, GameId.MTG]

        loss = dojo.compute_loss(embeddings, labels)
        loss.backward()

        assert loss.dim() == 0
        assert head.weight.grad is not None
