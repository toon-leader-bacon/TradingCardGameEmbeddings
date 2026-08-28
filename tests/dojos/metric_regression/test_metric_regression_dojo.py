import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import CardCount, Split
from src.dojos.losses.mse_loss import MSELoss
from src.dojos.metric_regression.metric_regression_dojo import MetricRegressionDojo
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_METRIC_NAME = "average_pick_number"


def _card(name: str) -> GenericCard:
    return GenericCard(
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


def _binder(num_cards: int) -> tuple[CardBinder, list[UUID]]:
    binder = CardBinder()
    uuids = []
    for i in range(num_cards):
        result = binder.add(_card(f"card-{i}"))
        uuids.append(result.stored_card.nocab_uuid)
    return binder, uuids


def _write_metrics_parquet(
    path: Path,
    rows: list[tuple[UUID, str, float]],
    expansion: str = "MSH",
    format_code: str = "PremierDraft",
) -> None:
    df = pd.DataFrame(
        [
            {
                "nocab_uuid": str(nocab_uuid),
                "metric_name": metric_name,
                "value": value,
                "sample_size": 10,
                "expansion": expansion,
                "format": format_code,
            }
            for nocab_uuid, metric_name, value in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _dojo(
    metrics_path: Path,
    metric_name: str = _METRIC_NAME,
    train_ratio: float = 0.5,
    validate_ratio: float = 0.2,
    rng_seed: int = 0,
) -> MetricRegressionDojo:
    return MetricRegressionDojo(
        metrics_path=metrics_path,
        metric_name=metric_name,
        train_ratio=train_ratio,
        validate_ratio=validate_ratio,
        loss=MSELoss(),
        rng_seed=rng_seed,
    )


class TestInit:
    @pytest.mark.parametrize("train_ratio", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_train_ratio_raises(self, train_ratio: float, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            MetricRegressionDojo(
                metrics_path=tmp_path / "metrics.parquet",
                metric_name=_METRIC_NAME,
                train_ratio=train_ratio,
                validate_ratio=0.1,
                loss=MSELoss(),
                rng_seed=0,
            )

    @pytest.mark.parametrize("validate_ratio", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_validate_ratio_raises(
        self, validate_ratio: float, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            MetricRegressionDojo(
                metrics_path=tmp_path / "metrics.parquet",
                metric_name=_METRIC_NAME,
                train_ratio=0.5,
                validate_ratio=validate_ratio,
                loss=MSELoss(),
                rng_seed=0,
            )

    def test_ratios_summing_to_one_or_more_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            MetricRegressionDojo(
                metrics_path=tmp_path / "metrics.parquet",
                metric_name=_METRIC_NAME,
                train_ratio=0.5,
                validate_ratio=0.5,
                loss=MSELoss(),
                rng_seed=0,
            )


class TestCardCount:
    def test_is_single_card(self, tmp_path: Path) -> None:
        assert _dojo(tmp_path / "metrics.parquet").card_count() == CardCount(minimum=1, maximum=1)


class TestBeforePrepareSplits:
    def test_split_size_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            _dojo(tmp_path / "metrics.parquet").split_size(Split.TRAIN)

    def test_next_batch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            _dojo(tmp_path / "metrics.parquet").next_batch(Split.TRAIN, batch_size=1)


class TestPrepareSplits:
    def test_splits_partition_every_labeled_card_without_gaps(self, tmp_path: Path) -> None:
        binder, uuids = _binder(20)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path, [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        )
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)

        dojo.prepare_splits(binder, held_out_cards=set())

        total = (
            dojo.split_size(Split.TRAIN)
            + dojo.split_size(Split.TEST)
            + dojo.split_size(Split.VALIDATE)
        )
        assert total == 20

    def test_only_matching_metric_name_rows_are_used(self, tmp_path: Path) -> None:
        binder, uuids = _binder(10)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path,
            [(u, _METRIC_NAME, 1.0) for u in uuids[:6]]
            + [(u, "pick_sideboard_rate", 0.5) for u in uuids[6:]],
        )
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)

        dojo.prepare_splits(binder, held_out_cards=set())

        total = (
            dojo.split_size(Split.TRAIN)
            + dojo.split_size(Split.TEST)
            + dojo.split_size(Split.VALIDATE)
        )
        assert total == 6

    def test_held_out_cards_never_appear_in_train_draws(self, tmp_path: Path) -> None:
        binder, uuids = _binder(10)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path, [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        )
        held_out_uuid = uuids[0]
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)

        dojo.prepare_splits(binder, held_out_cards={held_out_uuid})

        for _ in range(20):
            batch = dojo.next_batch(Split.TRAIN, batch_size=dojo.split_size(Split.TRAIN) or 1)
            drawn_uuids = {cs.cards[0].nocab_uuid for cs in batch.card_sets}
            assert held_out_uuid not in drawn_uuids

    def test_metric_uuid_missing_from_corpus_raises(self, tmp_path: Path) -> None:
        binder, _ = _binder(3)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(metrics_path, [(uuid4(), _METRIC_NAME, 1.0)])
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)

        with pytest.raises(ValueError):
            dojo.prepare_splits(binder, held_out_cards=set())


class TestNextBatchTrain:
    def test_returns_requested_batch_size(self, tmp_path: Path) -> None:
        binder, uuids = _binder(20)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path, [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        )
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())

        batch = dojo.next_batch(Split.TRAIN, batch_size=5)

        assert len(batch.card_sets) == 5
        assert len(batch.labels) == 5

    def test_each_card_set_holds_exactly_one_card(self, tmp_path: Path) -> None:
        binder, uuids = _binder(20)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path, [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        )
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())

        batch = dojo.next_batch(Split.TRAIN, batch_size=3)

        for card_set in batch.card_sets:
            assert len(card_set.cards) == 1

    def test_labels_match_each_card_own_metric_value(self, tmp_path: Path) -> None:
        binder, uuids = _binder(20)
        metrics_path = tmp_path / "metrics.parquet"
        rows = [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        _write_metrics_parquet(metrics_path, rows)
        expected = {u: v for u, _, v in rows}
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())

        batch = dojo.next_batch(Split.TRAIN, batch_size=5)

        for card_set, label in zip(batch.card_sets, batch.labels):
            assert label == expected[card_set.cards[0].nocab_uuid]


@pytest.mark.parametrize("split", [Split.TEST, Split.VALIDATE])
class TestNextBatchExhaustivePasses:
    def test_one_full_pass_covers_every_card_exactly_once(
        self, split: Split, tmp_path: Path
    ) -> None:
        binder, uuids = _binder(20)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path, [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        )
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)
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

    def test_final_batch_of_a_pass_is_short_not_padded(self, split: Split, tmp_path: Path) -> None:
        binder, uuids = _binder(20)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path, [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        )
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())
        pool_size = dojo.split_size(split)
        assert pool_size >= 2
        batch_size = pool_size - 1

        dojo.next_batch(split, batch_size)  # consumes pool_size - 1
        final_batch = dojo.next_batch(split, batch_size)  # only 1 left

        assert len(final_batch.card_sets) == 1

    def test_wraps_to_a_fresh_pass_after_full_coverage(self, split: Split, tmp_path: Path) -> None:
        binder, uuids = _binder(20)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(
            metrics_path, [(u, _METRIC_NAME, float(i)) for i, u in enumerate(uuids)]
        )
        dojo = _dojo(metrics_path, train_ratio=0.5, validate_ratio=0.2)
        dojo.prepare_splits(binder, held_out_cards=set())
        pool_size = dojo.split_size(split)
        batch_size = pool_size

        dojo.next_batch(split, batch_size)  # drains the whole pool
        next_pass_first_batch = dojo.next_batch(split, batch_size)

        assert len(next_pass_first_batch.card_sets) == pool_size


class TestBuildDecoderHead:
    def test_returns_linear_layer_with_one_output(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path / "metrics.parquet")

        head = dojo.build_decoder_head(embedding_dim=16)

        assert isinstance(head, torch.nn.Linear)
        assert head.in_features == 16
        assert head.out_features == 1


class TestComputeLoss:
    def test_before_build_decoder_head_raises(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path / "metrics.parquet")

        with pytest.raises(ValueError):
            dojo.compute_loss([[torch.randn(4)]], [1.0])

    def test_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path / "metrics.parquet")
        dojo.build_decoder_head(embedding_dim=4)

        with pytest.raises(ValueError):
            dojo.compute_loss([[torch.randn(4)]], [1.0, 2.0])

    def test_wrong_cards_per_example_raises(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path / "metrics.parquet")
        dojo.build_decoder_head(embedding_dim=4)

        with pytest.raises(ValueError):
            dojo.compute_loss([[torch.randn(4), torch.randn(4)]], [1.0])

    def test_returns_scalar_and_backpropagates(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path / "metrics.parquet")
        head = dojo.build_decoder_head(embedding_dim=4)
        embeddings = [[torch.randn(4, requires_grad=True)] for _ in range(3)]
        labels = [1.0, 2.5, 0.0]

        loss = dojo.compute_loss(embeddings, labels)
        loss.backward()

        assert loss.dim() == 0
        assert head.weight.grad is not None

    def test_handles_a_single_example_batch_without_collapsing_to_scalar(
        self, tmp_path: Path
    ) -> None:
        # Regression check for the squeeze(dim=1)-not-bare-squeeze() fix. A
        # bare .squeeze() on a batch of size 1 collapses (1, 1) all the way
        # to a 0-d tensor, but MSELoss's broadcasting mean-reduction would
        # still silently produce a 0-d loss either way — so this asserts on
        # the actual predictions shape handed to Loss.compute (via a spy),
        # not on the final loss's own dimensionality.
        class _SpyLoss:
            def __init__(self) -> None:
                self.seen_predictions_shape: torch.Size | None = None

            def compute(self, predictions: torch.Tensor, labels: list[float]) -> torch.Tensor:
                self.seen_predictions_shape = predictions.shape
                return predictions.sum()

        spy = _SpyLoss()
        dojo = MetricRegressionDojo(
            metrics_path=tmp_path / "metrics.parquet",
            metric_name=_METRIC_NAME,
            train_ratio=0.5,
            validate_ratio=0.2,
            loss=spy,
            rng_seed=0,
        )
        dojo.build_decoder_head(embedding_dim=4)
        embeddings = [[torch.randn(4, requires_grad=True)]]
        labels = [1.0]

        dojo.compute_loss(embeddings, labels)

        assert spy.seen_predictions_shape == (1,)
