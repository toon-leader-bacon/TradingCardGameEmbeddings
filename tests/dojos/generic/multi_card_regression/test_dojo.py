from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.batch import Batch
from src.dojos.dojo import BatchBudget
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.loss.huber_loss import HuberLoss
from src.dojos.loss.mse_loss import MseLoss
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.loss.standardized_label_loss import StandardizedLabelLoss
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import MultiCardInput, TrainingDatum

_BUDGET = BatchBudget(max_cost=1000, cost_of=lambda card: 1)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class _StubDataConstructor:
    """Builds one fixed-value TrainingDatum per row in a chunk, regardless
    of the row's actual content - isolates MultiCardRegressionDojo's own
    wiring (splits/batching/loss) from any particular DataConstructor."""

    def __init__(self, deck: MultiCardInput, label: float) -> None:
        self._deck = deck
        self._label = label

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        # Alternating labels: calibration rejects a constant regression label
        return [(self._deck, self._label + i % 2) for i in range(len(chunk))]


class _StubPooler:
    def pool(self, embeddings):
        return torch.zeros(embeddings[0].shape[0])


def _write_source(path: Path, num_rows: int) -> None:
    df = pd.DataFrame({"deck_uuid": [str(uuid4()) for _ in range(num_rows)]})
    df.to_parquet(path, index=False)


class TestMultiCardRegressionDojoSplits:
    def test_training_data_yields_batches_covering_all_rows(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        deck = [_card("Strike"), _card("Defend")]
        dojo = MultiCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(deck, 3.0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            config=DojoConfig(
                rng_seed=0,
                strict_version_check=False,
                output_directory=tmp_path / "splits",
            ),
        )

        batches = list(dojo.batches(Split.TRAIN, _BUDGET))

        total_rows = sum(len(batch.labels) for batch in batches)
        assert total_rows == 80  # 8/1/1 split of 100 rows
        assert all(isinstance(batch, Batch) for batch in batches)

    def test_test_and_validation_data_are_disjoint_from_training(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        deck = [_card("Strike"), _card("Defend")]
        dojo = MultiCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(deck, 3.0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            config=DojoConfig(
                rng_seed=0,
                strict_version_check=False,
                output_directory=tmp_path / "splits",
            ),
        )

        train_rows = sum(
            len(batch.labels) for batch in dojo.batches(Split.TRAIN, _BUDGET)
        )
        test_rows = sum(
            len(batch.labels) for batch in dojo.batches(Split.TEST, _BUDGET)
        )
        validation_rows = sum(
            len(batch.labels) for batch in dojo.batches(Split.VALIDATION, _BUDGET)
        )

        assert (train_rows, test_rows, validation_rows) == (80, 10, 10)


class TestMultiCardRegressionDojoPooler:
    def test_pooler_is_forwarded_to_decoder_head(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        deck = [_card("Strike")]
        stub_pooler = _StubPooler()

        dojo = MultiCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(deck, 1.0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            pooler=stub_pooler,
            config=DojoConfig(
                rng_seed=0,
                strict_version_check=False,
                output_directory=tmp_path / "splits",
            ),
        )

        assert dojo.decoder_head.pooler is stub_pooler


class TestMultiCardRegressionDojoComputeLoss:
    def test_returns_scalar_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        deck = [_card("Strike")]
        dojo = MultiCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(deck, 1.0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            config=DojoConfig(
                rng_seed=0,
                strict_version_check=False,
                output_directory=tmp_path / "splits",
            ),
        )
        # Two decks of different sizes - compute_loss must handle the
        # ragged BatchedMultiCardEmbedding shape correctly.
        embeddings = [[torch.randn(4), torch.randn(4)], [torch.randn(4)]]
        labels = [3.0, 5.0]

        loss = dojo.compute_loss(embeddings, Batch([deck] * len(labels), labels))

        assert loss.shape == ()

    def test_raises_on_mismatched_lengths(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        deck = [_card("Strike")]
        dojo = MultiCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(deck, 1.0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            config=DojoConfig(
                rng_seed=0,
                strict_version_check=False,
                output_directory=tmp_path / "splits",
            ),
        )

        try:
            dojo.compute_loss([[torch.randn(4)]], Batch([deck] * len([3, 5]), [3, 5]))
            assert False, "expected ValueError"
        except ValueError:
            pass


class _OutlierDataConstructor:
    """One 40-unit label among labels of 0 and 1: a heavy tail after z-scoring."""

    def __init__(self, deck: MultiCardInput) -> None:
        self._deck = deck

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        labels = [40.0 if i == 0 else float(i % 2) for i in range(len(chunk))]
        return [(self._deck, label) for label in labels]


class TestRegressionObjective:
    def _dojo(self, tmp_path: Path, objective: RegressionObjective | None):
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=400)
        return MultiCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_OutlierDataConstructor([_card("Strike")]),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            config=DojoConfig(
                rng_seed=0,
                strict_version_check=False,
                output_directory=tmp_path / "splits",
            ),
            objective=objective,
        )

    def _baseline(self, dojo) -> float:
        return dojo.baseline_loss(next(dojo.batches(Split.TRAIN, _BUDGET)))

    def test_defaults_to_mse_with_the_mean_predictors_baseline(
        self, tmp_path: Path
    ) -> None:
        dojo = self._dojo(tmp_path, None)

        assert isinstance(dojo.loss_calculator, StandardizedLabelLoss)
        assert isinstance(dojo.loss_calculator.inner, MseLoss)
        assert self._baseline(dojo) == 1.0

    def test_a_huber_objective_trains_with_huber_and_its_own_baseline(
        self, tmp_path: Path
    ) -> None:
        dojo = self._dojo(tmp_path, RegressionObjective.huber())

        assert isinstance(dojo.loss_calculator, StandardizedLabelLoss)
        assert isinstance(dojo.loss_calculator.inner, HuberLoss)
        # The outlier inflates the std, so Huber's baseline is well below 1
        assert 0.0 < self._baseline(dojo) < 0.9
