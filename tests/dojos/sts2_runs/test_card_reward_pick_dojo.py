from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import BatchBudget
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.sts2_runs.card_reward_pick_dojo import CardRewardPickDojo
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

# Wiring tests: placeholder calibration, no real TRAIN baseline fit
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")

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


def _write_source(path: Path, binder: CardBinder, cards: list[GenericCard]) -> None:
    """Twenty rows over two runs; every other row is a skip."""
    offered = [str(card.nocab_uuid) for card in cards[:3]]
    deck = [str(cards[3].nocab_uuid)]
    rows = 20
    pd.DataFrame(
        {
            "run_id": [f"run{i % 2}" for i in range(rows)],
            "deck_uuids": [deck] * rows,
            "offered_uuids": [offered] * rows,
            "picked_uuid": [None if i % 2 else offered[1] for i in range(rows)],
        }
    ).to_parquet(path, index=False)


def _dojo(tmp_path: Path) -> CardRewardPickDojo:
    binder = CardBinder()
    cards = [_card(name) for name in "ABCD"]
    for card in cards:
        binder.create(card)
    source = tmp_path / "source.parquet"
    _write_source(source, binder, cards)
    return CardRewardPickDojo(
        binder,
        HoldoutSpec.no_holdout(),
        card_embedding_size=4,
        path_to_training_data=source,
        rng_seed=0,
        strict_version_check=False,
    )


class TestCardRewardPickDojo:
    def test_wires_a_skippable_option_selection_cell(self, tmp_path: Path) -> None:
        dojo = _dojo(tmp_path)

        assert isinstance(dojo, MultiGroupOptionSelectionDojo)
        assert dojo.decoder_head.skip_embedding is not None

    def test_splits_by_run_so_no_run_straddles_train_and_test(
        self, tmp_path: Path
    ) -> None:
        # Two runs over an 8/1/1 split: the group split keeps each whole
        dojo = _dojo(tmp_path)

        counts = [
            sum(len(b.labels) for b in dojo.batches(split, _BUDGET))
            for split in (Split.TRAIN, Split.TEST, Split.VALIDATION)
        ]

        assert sum(counts) == 20
        assert all(count in (0, 10, 20) for count in counts)

    def test_every_label_fits_the_logits_the_head_produces(
        self, tmp_path: Path
    ) -> None:
        dojo = _dojo(tmp_path)
        batches = [
            batch
            for split in (Split.TRAIN, Split.TEST, Split.VALIDATION)
            for batch in dojo.batches(split, _BUDGET)
        ]
        assert batches

        for batch in batches:
            for example, label in zip(batch.inputs, batch.labels):
                offered, deck = example
                embeddings = [
                    [torch.randn(4) for _ in offered],
                    [torch.randn(4) for _ in deck],
                ]
                (logits,) = dojo.decoder_head([embeddings])
                # The skip is the extra option after the 3 offered cards
                assert logits.shape == (len(offered) + 1,)
                assert 0 <= label <= len(offered)
