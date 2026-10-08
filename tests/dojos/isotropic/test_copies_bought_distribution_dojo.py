from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import BatchBudget
from src.dojos.isotropic.copies_bought_distribution_dojo import (
    COPIES_LABEL_VALUES,
    CopiesBoughtDataConstructor,
    CopiesBoughtDistributionDojo,
    copies_label,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")


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


class TestCopiesLabel:
    @pytest.mark.parametrize(
        "copies,label", [(1, "1"), (9, "9"), (10, "10+"), (172, "10+")]
    )
    def test_buckets_the_tail_into_one_class(self, copies: int, label: str) -> None:
        assert copies_label(copies) == label
        assert label in COPIES_LABEL_VALUES

    def test_zero_copies_is_not_an_observation(self) -> None:
        assert copies_label(0) is None


class TestCopiesBoughtDataConstructor:
    def test_pairs_each_known_card_with_its_bucketed_count(self) -> None:
        binder = CardBinder()
        village = _card("Village")
        binder.create(village)
        chunk = pd.DataFrame(
            {
                "nocab_uuid": [
                    str(village.nocab_uuid),
                    str(uuid4()),
                    str(village.nocab_uuid),
                ],
                "copies": [3, 2, 40],
            }
        )

        result = CopiesBoughtDataConstructor().build(chunk, binder)

        assert result == [(village, "3"), (village, "10+")]


def test_the_dojo_trains_on_the_metrics_rows(tmp_path: Path) -> None:
    binder = CardBinder()
    village = _card("Village")
    binder.create(village)
    source = tmp_path / "copies_source.parquet"
    pd.DataFrame(
        {"nocab_uuid": [str(village.nocab_uuid)] * 10, "copies": [1, 2] * 5}
    ).to_parquet(source, index=False)

    dojo = CopiesBoughtDistributionDojo(
        binder,
        HoldoutSpec.no_holdout(),
        card_embedding_size=4,
        path_to_training_data=source,
        rng_seed=0,
        strict_version_check=False,
    )

    labels = {
        label
        for split in (Split.TRAIN, Split.TEST, Split.VALIDATION)
        for batch in dojo.batches(split, BatchBudget(max_cost=100, cost_of=lambda c: 1))
        for label in batch.labels
    }
    assert labels <= {"1", "2"}
    assert dojo.label_values == list(COPIES_LABEL_VALUES)
