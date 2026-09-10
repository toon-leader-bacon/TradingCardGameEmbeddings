from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd
import torch

from src.dojos.batch import Batch
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.type_hints import TrainingDatum


def _card(name: str, raw_content: dict | None = None) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name=name,
        raw_content=raw_content or {},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class _StubDataConstructor:
    """Builds one fixed-value TrainingDatum per row in a chunk, regardless
    of the row's actual content - isolates
    SingleCardFixedClassificationDojo's own wiring (splits/batching/loss)
    from any particular DataConstructor."""

    def __init__(self, card: GenericCard, label: str) -> None:
        self._card = card
        self._label = label

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        return [(self._card, self._label) for _ in range(len(chunk))]


def _write_source(path: Path, num_rows: int) -> None:
    df = pd.DataFrame({"nocab_uuid": [str(uuid4()) for _ in range(num_rows)]})
    df.to_parquet(path, index=False)


class TestSingleCardFixedClassificationDojoSplits:
    def test_training_data_yields_batches_covering_all_rows(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        card = _card("Geralt of Rivia")
        dojo = SingleCardFixedClassificationDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card, "monster"),
            label_values=["monster", "northern_realms"],
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
        card = _card("Geralt of Rivia")
        dojo = SingleCardFixedClassificationDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card, "monster"),
            label_values=["monster", "northern_realms"],
            card_embedding_size=4,
            rng_seed=0,
        )

        train_rows = sum(len(batch.labels) for batch in dojo.training_data())
        test_rows = sum(len(batch.labels) for batch in dojo.test_data())
        validation_rows = sum(len(batch.labels) for batch in dojo.validation_data())

        assert (train_rows, test_rows, validation_rows) == (80, 10, 10)


class TestSingleCardFixedClassificationDojoModPipeline:
    def test_masking_mod_applies_on_every_split_not_just_training(
        self, tmp_path: Path
    ) -> None:
        # MaskTargetKeyMod(train_only=False) is a structural requirement
        # for this task shape (see gwent_one/masked_field_dojos.py) - it
        # must fire on test/validation data too, unlike a plain
        # train_only augmentation mod, or the model could read the
        # answer straight off raw_content at eval time.
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        card = _card("Geralt of Rivia", raw_content={"faction": "monster"})
        dojo = SingleCardFixedClassificationDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card, "monster"),
            label_values=["monster", "northern_realms"],
            card_embedding_size=4,
            mod_pipeline=ModPipeline(
                [MaskTargetKeyMod(key="faction", train_only=False)]
            ),
            rng_seed=0,
        )

        train_batches = list(dojo.training_data())
        test_batches = list(dojo.test_data())
        validation_batches = list(dojo.validation_data())

        for batches in (train_batches, test_batches, validation_batches):
            for batch in batches:
                for input_card in batch.inputs:
                    assert input_card.raw_content["faction"] == "[MASK]"


class TestSingleCardFixedClassificationDojoComputeLoss:
    def test_returns_scalar_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        card = _card("Geralt of Rivia")
        dojo = SingleCardFixedClassificationDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card, "monster"),
            label_values=["monster", "northern_realms"],
            card_embedding_size=4,
            rng_seed=0,
        )
        embeddings = [torch.randn(4), torch.randn(4)]
        labels = ["monster", "northern_realms"]

        loss = dojo.compute_loss(embeddings, labels)

        assert loss.shape == ()

    def test_raises_on_mismatched_lengths(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        card = _card("Geralt of Rivia")
        dojo = SingleCardFixedClassificationDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card, "monster"),
            label_values=["monster", "northern_realms"],
            card_embedding_size=4,
            rng_seed=0,
        )

        try:
            dojo.compute_loss([torch.randn(4)], ["monster", "northern_realms"])
            assert False, "expected ValueError"
        except ValueError:
            pass


class _StubLoss:
    """Records the label_values it was built with, and always returns a
    fixed scalar - isolates the loss_factory injection point itself
    from any real loss's actual math."""

    def __init__(self, label_values: List[str]) -> None:
        self.label_values = label_values

    def calculate(self, decoder_output: object, labels: object) -> torch.Tensor:
        return torch.tensor(0.0)


class TestSingleCardFixedClassificationDojoLossFactory:
    def test_defaults_to_fixed_classification_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        card = _card("Geralt of Rivia")
        dojo = SingleCardFixedClassificationDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card, "monster"),
            label_values=["monster", "northern_realms"],
            card_embedding_size=4,
            rng_seed=0,
        )

        assert isinstance(dojo.loss_calculator, FixedClassificationLoss)

    def test_loss_factory_builds_loss_calculator_from_label_values(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        card = _card("Geralt of Rivia")
        dojo = SingleCardFixedClassificationDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(card, "monster"),
            label_values=["monster", "northern_realms"],
            card_embedding_size=4,
            loss_factory=_StubLoss,
            rng_seed=0,
        )

        assert isinstance(dojo.loss_calculator, _StubLoss)
        assert dojo.loss_calculator.label_values == ["monster", "northern_realms"]
