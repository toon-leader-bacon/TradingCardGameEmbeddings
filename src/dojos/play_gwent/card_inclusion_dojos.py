"""Thin wrappers over the Gwent card inclusion metrics
(src/data_refinement/metrics/play_gwent/card_inclusion_metrics.py), the
twins of ../fabtcg_decklists/card_inclusion_dojos.py (see there for why
FactionConditionedInclusionDojo reuses PickNumberDecayCurveDataConstructor
behind isotropic's column-renaming Decorator).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.play_gwent.card_inclusion_metrics import (
    CardInclusionRateMetric,
    FactionConditionedInclusionMetric,
)
from src.dojos.generic.data_constructors import PickNumberDecayCurveDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.generic.single_card_fixed_classification.loss_spec import (
    MASKED_VECTOR_REGRESSION_LOSS_SPEC,
)
from src.dojos.generic.renamed_column_data_constructor import (
    RenamedColumnDataConstructor,
)
from src.schema.holdout import HoldoutSpec

# A faction position needs this many of the faction's decks (every
# faction has thousands, so today nothing is dropped)
MIN_FACTION_DECKS = 20


class CardInclusionRateDojo(CardAverageMetricDojo):
    """Card -> P(card in deck | card legal for the deck's faction)."""

    METRIC = CardInclusionRateMetric


class FactionConditionedInclusionDojo(SingleCardFixedClassificationDojo):
    """Card -> P(card in deck | faction) per GWENT_FACTIONS faction;
    positions where the card is illegal (count 0) are masked out of the
    loss, so a faction card trains one position and a neutral card six."""

    METRIC = FactionConditionedInclusionMetric

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """See PickNumberDecayCurveDojo; the label vector is one position
        per FactionConditionedInclusionMetric.LABEL_VALUES faction."""
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=RenamedColumnDataConstructor(
                PickNumberDecayCurveDataConstructor(min_sample_count=MIN_FACTION_DECKS),
                {
                    "inclusion_rate_by_faction": "take_rate_by_pick_number",
                    "legal_deck_count_by_faction": "sample_count_by_pick_number",
                },
            ),
            label_values=self.METRIC.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            loss_spec=MASKED_VECTOR_REGRESSION_LOSS_SPEC,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
