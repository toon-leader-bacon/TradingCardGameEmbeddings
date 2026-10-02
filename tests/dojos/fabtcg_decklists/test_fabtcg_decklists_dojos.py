from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.fabtcg_decklists.card_inclusion_metrics import (
    CardInclusionRateMetric,
    HeroConditionedInclusionMetric,
)
from src.data_refinement.metrics.fabtcg_decklists.hero_masked_from_deck_metric import (
    HeroMaskedFromDeckMetric,
)
from src.dojos.fabtcg_decklists.card_inclusion_dojos import (
    CardInclusionRateDojo,
    HeroConditionedInclusionDojo,
)
from src.dojos.fabtcg_decklists.deck_card_mask_dojos import HeroMaskedFromDeckDojo
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss
from src.schema.holdout import HoldoutSpec
from tests.data_refinement.metrics.fabtcg_decklists.fab_fixtures import (
    fab_binder,
    fab_card,
)

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")


def _write(path: Path, columns: dict) -> Path:
    pd.DataFrame(columns).to_parquet(path, index=False)
    return path


class TestHeroMaskedFromDeckDojo:
    def test_wires_deck_card_mask_constructor_and_hero_labels(
        self, tmp_path: Path
    ) -> None:
        source = _write(
            tmp_path / "hero.parquet",
            {
                "deck_uuid": ["00000000-0000-0000-0000-000000000000"] * 10,
                "target_card_uuid": ["00000000-0000-0000-0000-000000000001"] * 10,
                "label": ["Dorinthea Ironsong"] * 10,
            },
        )
        deck_box = DeckBox()

        dojo = HeroMaskedFromDeckDojo(
            CardBinder(),
            HoldoutSpec.no_holdout(),
            deck_box,
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )

        assert isinstance(dojo, MultiCardFixedClassificationDojo)
        assert isinstance(dojo.data_constructor, DeckCardMaskDataConstructor)
        assert dojo.label_values == list(HeroMaskedFromDeckMetric.LABEL_VALUES)
        assert HeroMaskedFromDeckDojo.METRIC is HeroMaskedFromDeckMetric


class TestCardInclusionRateDojo:
    def test_reads_the_metric_label_column(self) -> None:
        assert CardInclusionRateDojo.METRIC is CardInclusionRateMetric
        assert CardInclusionRateMetric.LABEL_COLUMN == "inclusion_rate"


class TestHeroConditionedInclusionDojo:
    def test_builds_masked_rate_vectors_from_metric_rows(self, tmp_path: Path) -> None:
        card = fab_card("Sink Below", "Generic Defense Reaction")
        width = len(HeroConditionedInclusionMetric.LABEL_VALUES)
        rates = [None] * width
        counts = [0] * width
        rates[0], counts[0] = 0.25, 40
        rates[1], counts[1] = 0.5, 10  # under MIN_HERO_DECKS: masked
        source = _write(
            tmp_path / "by_hero.parquet",
            {
                "nocab_uuid": [str(card.nocab_uuid)] * 10,
                "inclusion_rate_by_hero": [rates] * 10,
                "legal_deck_count_by_hero": [counts] * 10,
            },
        )
        binder = fab_binder([card])

        dojo = HeroConditionedInclusionDojo(
            binder,
            HoldoutSpec.no_holdout(),
            card_embedding_size=4,
            path_to_training_data=source,
            rng_seed=0,
            strict_version_check=False,
        )
        built = dojo.data_constructor.build(pd.read_parquet(source).head(1), binder)

        assert dojo.label_values == list(HeroConditionedInclusionMetric.LABEL_VALUES)
        assert isinstance(dojo.loss_calculator, MaskedVectorRegressionLoss)
        assert built == [(card, {0: 0.25})]
