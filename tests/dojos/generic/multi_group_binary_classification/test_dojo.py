import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd
import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.batch import Batch
from src.dojos.dojo import BatchBudget
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_group_binary_classification.dojo import (
    MultiGroupBinaryClassificationDojo,
)
from src.dojos.generic.multi_group_binary_classification.group_swap_mod import (
    GroupSwapMod,
)
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import MultiGroupInput, TrainingDatum

_BUDGET = BatchBudget(max_cost=1000, cost_of=lambda card: 1)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.DOMINION,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.DOMINIONTABS,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class _StubDataConstructor:
    """One datum per row, alternating labels 0.0/1.0 (a one-class sample
    would have a zero binary-entropy baseline)."""

    def __init__(self, group: MultiGroupInput) -> None:
        self._group = group

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        return [(self._group, float(i % 2)) for i in range(len(chunk))]


def _dojo(
    tmp_path: Path,
    group: MultiGroupInput,
    num_rows: int = 100,
    mod_pipeline: ModPipeline | None = None,
) -> MultiGroupBinaryClassificationDojo:
    source = tmp_path / "source.parquet"
    pd.DataFrame({"row_id": [str(uuid4()) for _ in range(num_rows)]}).to_parquet(
        source, index=False
    )
    return MultiGroupBinaryClassificationDojo(
        path_to_training_data=source,
        data_constructor=_StubDataConstructor(group),
        card_lookup=CardBinder(),
        holdout=HoldoutSpec.no_holdout(),
        card_embedding_size=4,
        mod_pipeline=mod_pipeline,
        config=DojoConfig(
            rng_seed=0,
            strict_version_check=False,
            output_directory=tmp_path / "splits",
        ),
    )


class TestSplits:
    def test_splits_cover_every_row(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path, [[_card("A")], [_card("B")]])

        counts = [
            sum(len(batch.labels) for batch in dojo.batches(split, _BUDGET))
            for split in (Split.TRAIN, Split.TEST, Split.VALIDATION)
        ]

        assert counts == [80, 10, 10]

    def test_empty_second_group_is_accepted(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path, [[_card("A")], []])

        assert sum(len(b.labels) for b in dojo.batches(Split.TRAIN, _BUDGET)) == 80


class TestCalibration:
    def test_baseline_is_binary_entropy_of_positive_rate(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path, [[_card("A")], [_card("B")]])
        batch = next(iter(dojo.batches(Split.TRAIN, _BUDGET)))

        # Alternating labels: positive rate 0.5 -> ln 2
        assert dojo.baseline_loss(batch) == pytest.approx(math.log(2), abs=0.01)


class TestComputeLoss:
    def test_returns_scalar_loss(self, tmp_path: Path) -> None:
        group = [[_card("A")], [_card("B")]]
        dojo = _dojo(tmp_path, group, num_rows=10)
        embeddings = [
            [[torch.randn(4), torch.randn(4)], [torch.randn(4)]],
            [[torch.randn(4)], []],
        ]

        loss = dojo.compute_loss(embeddings, Batch([group, group], [1.0, 0.0]))

        assert loss.shape == ()

    def test_raises_on_mismatched_lengths(self, tmp_path: Path) -> None:
        group = [[_card("A")], [_card("B")]]
        dojo = _dojo(tmp_path, group, num_rows=10)

        with pytest.raises(ValueError):
            dojo.compute_loss(
                [[[torch.randn(4)], [torch.randn(4)]]],
                Batch([group, group], [0.0, 1.0]),
            )


class TestGroupSwapMod:
    def test_swap_runs_on_train_only(self, tmp_path: Path) -> None:
        a, b = _card("A"), _card("B")
        dojo = _dojo(
            tmp_path,
            [[a], [b]],
            mod_pipeline=ModPipeline([GroupSwapMod(1.0)]),
        )

        train = next(iter(dojo.batches(Split.TRAIN, _BUDGET)))
        test = next(iter(dojo.batches(Split.TEST, _BUDGET)))

        assert all(inputs[0][0] is b for inputs in train.inputs)
        assert all(inputs[0][0] is a for inputs in test.inputs)
