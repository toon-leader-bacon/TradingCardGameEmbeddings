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
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
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
    of the row's actual content - isolates MultiGroupOptionSelectionDojo's
    own wiring (splits/batching/loss) from any particular DataConstructor."""

    def __init__(self, group: MultiGroupInput, pick_index: int) -> None:
        self._group = group
        self._pick_index = pick_index

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        return [(self._group, self._pick_index) for _ in range(len(chunk))]


class _StubScoringHead:
    def score(self, option_embeddings, context):
        return torch.zeros(len(option_embeddings))


class _StubPooler:
    def pool(self, embeddings):
        return torch.zeros(embeddings[0].shape[0])


def _write_source(path: Path, num_rows: int) -> None:
    df = pd.DataFrame({"draft_id": [str(uuid4()) for _ in range(num_rows)]})
    df.to_parquet(path, index=False)


class TestMultiGroupOptionSelectionDojoSplits:
    def test_training_data_yields_batches_covering_all_rows(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        group = [[_card("A"), _card("B")], [_card("C")]]
        dojo = MultiGroupOptionSelectionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(group, 1),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            rng_seed=0,
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
        group = [[_card("A")], []]  # empty pool group - must not break splitting
        dojo = MultiGroupOptionSelectionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(group, 0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            rng_seed=0,
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


class TestMultiGroupOptionSelectionDojoScoringHeadAndPooler:
    def test_scoring_head_is_forwarded_to_decoder_head(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        group = [[_card("A")], []]
        stub_scoring_head = _StubScoringHead()

        dojo = MultiGroupOptionSelectionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(group, 0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            scoring_head=stub_scoring_head,
            rng_seed=0,
        )

        assert dojo.decoder_head.scoring_head is stub_scoring_head

    def test_pooler_is_forwarded_to_decoder_head(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        group = [[_card("A")], []]
        stub_pooler = _StubPooler()

        dojo = MultiGroupOptionSelectionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(group, 0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            pooler=stub_pooler,
            rng_seed=0,
        )

        assert dojo.decoder_head.pooler is stub_pooler


class TestMultiGroupOptionSelectionDojoComputeLoss:
    def test_returns_scalar_loss(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        group = [[_card("A")], []]
        dojo = MultiGroupOptionSelectionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(group, 0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            rng_seed=0,
        )
        # One example with an empty pool, one with a non-empty pool -
        # compute_loss must handle both without raising.
        embeddings = [
            [[torch.randn(4), torch.randn(4)], []],
            [[torch.randn(4)], [torch.randn(4), torch.randn(4)]],
        ]
        labels = [1, 0]

        loss = dojo.compute_loss(embeddings, Batch([group] * len(labels), labels))

        assert loss.shape == ()

    def test_raises_on_mismatched_lengths(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        group = [[_card("A")], []]
        dojo = MultiGroupOptionSelectionDojo(
            path_to_training_data=source,
            data_constructor=_StubDataConstructor(group, 0),
            card_lookup=CardBinder(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            rng_seed=0,
        )

        with pytest.raises(ValueError):
            dojo.compute_loss(
                [[[torch.randn(4)], []]], Batch([group] * len([0, 1]), [0, 1])
            )
