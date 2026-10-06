"""Thin wrappers over the FaB card inclusion metrics
(src/data_refinement/metrics/fabtcg_decklists/card_inclusion_metrics.py).

HeroConditionedInclusionDojo follows PickNumberDecayCurveDojo: a vector
of independent rates (one per hero) with absent positions masked,
scored by MASKED_VECTOR_REGRESSION_LOSS_SPEC on the single-card fixed
classification cell. Its rows have the same shape as
PickNumberDecayCurveMetric's (a rate list and a parallel count list), so
PickNumberDecayCurveDataConstructor is reused behind isotropic's
column-renaming Decorator rather than copied.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.fabtcg_decklists.card_inclusion_metrics import (
    CardInclusionRateMetric,
    HeroConditionedInclusionMetric,
)
from src.data_refinement.metrics.fabtcg_decklists.hero_labels import (
    MIN_DECKS_PER_HERO,
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

# A hero position needs as many of the hero's decks as HERO_NAMES itself
# requires (so no labeled hero is masked for lack of decks)
MIN_HERO_DECKS = MIN_DECKS_PER_HERO


class CardInclusionRateDojo(CardAverageMetricDojo):
    """Card -> P(card in deck | card legal for the deck's hero)."""

    METRIC = CardInclusionRateMetric


class HeroConditionedInclusionDojo(SingleCardFixedClassificationDojo):
    """Card -> P(card in deck | hero) per HERO_NAMES hero; positions where
    the card is illegal (count 0) or the hero has fewer than
    MIN_HERO_DECKS decks are masked out of the loss."""

    METRIC = HeroConditionedInclusionMetric

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
        per HeroConditionedInclusionMetric.LABEL_VALUES hero."""
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=RenamedColumnDataConstructor(
                PickNumberDecayCurveDataConstructor(min_sample_count=MIN_HERO_DECKS),
                {
                    "inclusion_rate_by_hero": "take_rate_by_pick_number",
                    "legal_deck_count_by_hero": "sample_count_by_pick_number",
                },
            ),
            label_values=self.METRIC.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            loss_spec=MASKED_VECTOR_REGRESSION_LOSS_SPEC,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
