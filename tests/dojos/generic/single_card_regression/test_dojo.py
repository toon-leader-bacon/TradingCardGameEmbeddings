from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd
import torch

from src.dojos.batch import Batch
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.type_hints import TrainingDatum


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
    of the row's actual content - isolates SingleCardRegressionDojo's own
    wiring (splits/batching/loss) from any particular DataConstructor."""

    def __init__(self, card: GenericCard) -> None:
        self._card = card

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        return [(self._card, 1.0) for _ in range(len(chunk))]


def _write_source(path: Path, num_rows: int) -> None:
    df = pd.DataFrame({"nocab_uuid": [str(uuid4()) for _ in range(num_rows)]})
    df.to_parquet(path, index=False)


class TestSingleCardRegressionDojoSplits:
    def test_training_data_yields_batches_covering_all_rows(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        card = _card("Strike")
        dojo = SingleCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card),
            card_embedding_size=4,
            rng_seed=0,
        )

        batches = list(dojo.training_data())

        total_rows = sum(len(batch.labels) for batch in batches)
        assert total_rows == 80  # 8/1/1 split of 100 rows
        assert all(isinstance(batch, Batch) for batch in batches)

    def test_test_and_validation_data_are_disjoint_from_training(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        card = _card("Strike")
        dojo = SingleCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card),
            card_embedding_size=4,
            rng_seed=0,
        )

        train_rows = sum(len(batch.labels) for batch in dojo.training_data())
        test_rows = sum(len(batch.labels) for batch in dojo.test_data())
        validation_rows = sum(len(batch.labels) for batch in dojo.validation_data())

        assert (train_rows, test_rows, validation_rows) == (80, 10, 10)


class TestSingleCardRegressionDojoComputeLoss:
    def test_returns_scalar_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        card = _card("Strike")
        dojo = SingleCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card),
            card_embedding_size=4,
            rng_seed=0,
        )
        embeddings = [torch.randn(4), torch.randn(4)]
        labels = [1.0, 2.0]

        loss = dojo.compute_loss(embeddings, labels)

        assert loss.shape == ()

    def test_raises_on_mismatched_lengths(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        card = _card("Strike")
        dojo = SingleCardRegressionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card),
            card_embedding_size=4,
            rng_seed=0,
        )

        try:
            dojo.compute_loss([torch.randn(4)], [1.0, 2.0])
            assert False, "expected ValueError"
        except ValueError:
            pass
