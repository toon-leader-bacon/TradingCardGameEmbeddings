from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import BatchBudget
from src.dojos.sts2_runs.card_removal_pick_dojo import CardRemovalPickDojo
from src.dojos.sts2_runs.card_upgrade_pick_dojo import CardUpgradePickDojo
from src.dojos.sts2_runs.shop_purchase_pick_dojo import ShopPurchasePickDojo
from src.dojos.sts2_runs.sts2_run_pick_dojo import Sts2RunPickDojo
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


def _dojo(dojo_class: type[Sts2RunPickDojo], tmp_path: Path) -> Sts2RunPickDojo:
    binder = CardBinder()
    cards = [_card(name) for name in "ABCD"]
    for card in cards:
        binder.create(card)
    offered = [str(card.nocab_uuid) for card in cards[:3]]
    source = tmp_path / "source.parquet"
    rows = 20
    pd.DataFrame(
        {
            "run_id": [f"run{i % 2}" for i in range(rows)],
            "deck_uuids": [[str(cards[3].nocab_uuid)]] * rows,
            "offered_uuids": [offered] * rows,
            "picked_uuid": [offered[i % 3] for i in range(rows)],
        }
    ).to_parquet(source, index=False)
    return dojo_class(
        binder,
        HoldoutSpec.no_holdout(),
        card_embedding_size=4,
        path_to_training_data=source,
        rng_seed=0,
        strict_version_check=False,
    )


@pytest.mark.parametrize(
    "dojo_class", [ShopPurchasePickDojo, CardRemovalPickDojo, CardUpgradePickDojo]
)
class TestPickDojosWithoutASkip:
    def test_has_no_skip_option(
        self, dojo_class: type[Sts2RunPickDojo], tmp_path: Path
    ) -> None:
        assert _dojo(dojo_class, tmp_path).decoder_head.skip_embedding is None

    def test_every_label_is_one_of_the_offered_cards(
        self, dojo_class: type[Sts2RunPickDojo], tmp_path: Path
    ) -> None:
        dojo = _dojo(dojo_class, tmp_path)
        batches = [
            batch
            for split in (Split.TRAIN, Split.TEST, Split.VALIDATION)
            for batch in dojo.batches(split, _BUDGET)
        ]
        assert batches

        for batch in batches:
            for (offered, deck), label in zip(batch.inputs, batch.labels):
                embeddings = [
                    [torch.randn(4) for _ in offered],
                    [torch.randn(4) for _ in deck],
                ]
                (logits,) = dojo.decoder_head([embeddings])
                assert logits.shape == (len(offered),)
                assert 0 <= label < len(offered)

    def test_splits_by_run_so_no_run_straddles_train_and_test(
        self, dojo_class: type[Sts2RunPickDojo], tmp_path: Path
    ) -> None:
        # Two runs over an 8/1/1 split: the group split keeps each whole
        dojo = _dojo(dojo_class, tmp_path)

        counts = [
            sum(len(b.labels) for b in dojo.batches(split, _BUDGET))
            for split in (Split.TRAIN, Split.TEST, Split.VALIDATION)
        ]

        assert sum(counts) == 20
        assert all(count in (0, 10, 20) for count in counts)
