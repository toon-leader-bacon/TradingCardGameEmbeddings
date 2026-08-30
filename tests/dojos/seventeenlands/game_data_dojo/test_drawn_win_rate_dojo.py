from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.metric_scanner import MetricScanner
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.drawn_win_rate import (
    DrawnWinRateMetric,
)
from src.dojos.dojo import CardCount, Split
from src.dojos.seventeenlands.game_data_dojo.drawn_win_rate_dojo import DrawnWinRateDojo
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# DrawnWinRateDojo only wires MetricRegressionDojo's generic constructor
# to this one metric — its split/cursor/loss mechanics are already
# covered by tests/dojos/seventeenlands/test_metric_regression_dojo.py.
# These tests only cover what's actually new here: metric_name/
# metrics_path/loss defaulting, plus a smoke test that the wired-up
# dojo still satisfies the Dojo contract end to end.


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


def _write_metrics_parquet(path: Path, rows: list[tuple[UUID, float]]) -> None:
    df = pd.DataFrame(
        [
            {
                "nocab_uuid": str(nocab_uuid),
                "metric_name": DrawnWinRateMetric.name,
                "value": value,
                "sample_size": 10,
                "expansion": "MSH",
                "format": "PremierDraft",
            }
            for nocab_uuid, value in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


class TestInit:
    def test_defaults_metric_name_to_the_metric_class_own_name(self, tmp_path: Path) -> None:
        dojo = DrawnWinRateDojo(
            expansion="MSH",
            format_code="PremierDraft",
            train_ratio=0.5,
            validate_ratio=0.2,
            rng_seed=0,
            metrics_path=tmp_path / "metrics.parquet",
        )

        assert dojo._metric_name == DrawnWinRateMetric.name

    def test_defaults_metrics_path_via_metric_scanner_convention(self) -> None:
        dojo = DrawnWinRateDojo(
            expansion="MSH",
            format_code="PremierDraft",
            train_ratio=0.5,
            validate_ratio=0.2,
            rng_seed=0,
        )

        assert dojo._metrics_path == MetricScanner.default_output_path("MSH", "PremierDraft")

    def test_explicit_metrics_path_overrides_the_default(self, tmp_path: Path) -> None:
        explicit_path = tmp_path / "somewhere_else.parquet"

        dojo = DrawnWinRateDojo(
            expansion="MSH",
            format_code="PremierDraft",
            train_ratio=0.5,
            validate_ratio=0.2,
            rng_seed=0,
            metrics_path=explicit_path,
        )

        assert dojo._metrics_path == explicit_path


class TestEndToEnd:
    def test_satisfies_the_dojo_contract(self, tmp_path: Path) -> None:
        binder, uuids = _binder(10)
        metrics_path = tmp_path / "metrics.parquet"
        _write_metrics_parquet(metrics_path, [(u, float(i)) for i, u in enumerate(uuids)])
        dojo = DrawnWinRateDojo(
            expansion="MSH",
            format_code="PremierDraft",
            train_ratio=0.5,
            validate_ratio=0.2,
            rng_seed=0,
            metrics_path=metrics_path,
        )

        assert dojo.card_count() == CardCount(minimum=1, maximum=1)

        dojo.prepare_splits(binder, held_out_cards=set())
        batch = dojo.next_batch(Split.TRAIN, batch_size=3)

        assert len(batch.card_sets) == 3
        assert len(batch.labels) == 3

        dojo.build_decoder_head(embedding_dim=4)
        embeddings = [[torch.randn(4, requires_grad=True)] for _ in range(3)]
        loss = dojo.compute_loss(embeddings, batch.labels)
        assert loss.dim() == 0
