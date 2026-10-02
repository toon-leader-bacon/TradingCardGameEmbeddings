from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.play_gwent.card_inclusion_metrics import (
    CardInclusionRateMetric,
    FactionConditionedInclusionMetric,
)
from src.data_refinement.metrics.play_gwent.guide_votes_metric import (
    GuideVotesMetric,
)
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss
from src.dojos.play_gwent.card_inclusion_dojos import (
    CardInclusionRateDojo,
    FactionConditionedInclusionDojo,
)
from src.dojos.play_gwent.deck_label_dojos import GuideVotesDojo
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")


class TestCardInclusionRateDojo:
    def test_reads_the_metric_label_column(self) -> None:
        assert CardInclusionRateDojo.METRIC is CardInclusionRateMetric
        assert CardInclusionRateMetric.LABEL_COLUMN == "inclusion_rate"


class TestFactionConditionedInclusionDojo:
    def test_builds_masked_rate_vectors_from_metric_rows(self, tmp_path: Path) -> None:
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.GWENT,
            name="Tactical Advantage",
            raw_content={"name": "Tactical Advantage", "faction": "neutral"},
            provenance=Provenance(
                data_source=DataSource.GWENT_ONE,
                source_id="1",
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder = CardBinder()
        binder.create(card)
        source = tmp_path / "by_faction.parquet"
        pd.DataFrame(
            {
                "nocab_uuid": [str(card.nocab_uuid)] * 10,
                "inclusion_rate_by_faction": [[0.5, 0.25, None, None, None, None]] * 10,
                "legal_deck_count_by_faction": [[100, 5, 0, 0, 0, 0]] * 10,
            }
        ).to_parquet(source, index=False)

        dojo = FactionConditionedInclusionDojo(
            binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )
        built = dojo.data_constructor.build(pd.read_parquet(source).head(1), binder)

        assert dojo.label_values == list(FactionConditionedInclusionMetric.LABEL_VALUES)
        assert isinstance(dojo.loss_calculator, MaskedVectorRegressionLoss)
        # The 5-deck position is under MIN_FACTION_DECKS: masked
        assert built == [(card, {0: 0.5})]


class TestGuideVotesDojo:
    def test_wires_a_deck_label_regression_over_the_box(self, tmp_path: Path) -> None:
        source = tmp_path / "votes.parquet"
        pd.DataFrame(
            {
                "guide_id": list(range(10)),
                "deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
                GuideVotesMetric.LABEL_COLUMN: [0.5] * 10,
            }
        ).to_parquet(source, index=False)

        dojo = GuideVotesDojo(
            CardBinder(),
            HoldoutSpec.no_holdout(),
            DeckBox(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, MultiCardRegressionDojo)
        assert isinstance(dojo.data_constructor, DeckLabelDataConstructor)
        assert GuideVotesDojo.METRIC is GuideVotesMetric
