"""Thin per-metric wrappers over DeckLabelMetric's int64-labeled
concrete subclasses (src/data_refinement/metrics/sts_gg/deck_label_metrics.py).

SCOPE: the 8 int64-labeled DeckLabelMetric subclasses are wrapped as
DeckLabelMetricDojo subclasses (../generic/paired_metric_dojos.py) (RelicCountMetric,
TotalDamageTakenMetric, TotalCardsPickedMetric, TotalCardsSkippedMetric,
TotalTurnsMetric, ElitesKilledMetric, FloorsClearedMetric,
TotalCombatsMetric); WinMetric (bool) is wrapped as a
MultiCardBinaryClassificationDojo subclass (WinDojo); CharacterPrediction-
Metric (str, closed vocabulary) is wrapped as a
MultiCardFixedClassificationDojo subclass (CharacterDojo). Same
DeckLabelDataConstructor for all three shapes - its label_caster
defaults to float (fine for WinDojo's bool column, True/False ->
1.0/0.0) but CharacterDojo passes label_caster=str, since
FixedClassificationLoss needs the raw string label, not a float (see
data_constructors.py's DeckLabelDataConstructor docstring). No new
DataConstructor was needed for any of the three. KilledByMetric is
skipped entirely - flagged as dead data upstream (deck_label_metrics.py's
own module docstring).

NAMING: every wrapper below is Deck-prefixed (e.g. DeckRelicCountDojo)
to avoid colliding with src/dojos/sts_gg/card_average_dojos.py's
already-implemented CardRelicCountDojo and similar - CardAverageMetric
(single-card regression) and DeckLabelMetric (whole-deck regression)
are distinct metric families that happen to share topics like "relic
count," with distinct wrappers targeting different generic dojo cells.

NO POOLER/MOD_PIPELINE OVERRIDE: unlike gwent_one's masking mods, none
of these 8 metrics have a structural masking requirement, so every
wrapper leaves MultiCardRegressionDojo's mod_pipeline and pooler at
their defaults (empty pipeline, MeanEmbeddingPooler).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
    ElitesKilledMetric,
    FloorsClearedMetric,
    RelicCountMetric,
    TotalCardsPickedMetric,
    TotalCardsSkippedMetric,
    TotalCombatsMetric,
    TotalDamageTakenMetric,
    TotalTurnsMetric,
    WinMetric,
)
from src.dojos.generic.paired_metric_dojos import DeckLabelMetricDojo
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.schema.holdout import HoldoutSpec


class DeckRelicCountDojo(DeckLabelMetricDojo):
    """Deck -> final relicCount (RelicCountMetric)."""

    METRIC = RelicCountMetric


class DeckTotalDamageTakenDojo(DeckLabelMetricDojo):
    """Deck -> stats.totalDamageTaken (TotalDamageTakenMetric)."""

    METRIC = TotalDamageTakenMetric


class DeckTotalCardsPickedDojo(DeckLabelMetricDojo):
    """Deck -> stats.totalCardsPicked (TotalCardsPickedMetric)."""

    METRIC = TotalCardsPickedMetric


class DeckTotalCardsSkippedDojo(DeckLabelMetricDojo):
    """Deck -> stats.totalCardsSkipped (TotalCardsSkippedMetric)."""

    METRIC = TotalCardsSkippedMetric


class DeckTotalTurnsDojo(DeckLabelMetricDojo):
    """Deck -> stats.totalTurns (TotalTurnsMetric)."""

    METRIC = TotalTurnsMetric


class DeckElitesKilledDojo(DeckLabelMetricDojo):
    """Deck -> stats.elitesKilled (ElitesKilledMetric)."""

    METRIC = ElitesKilledMetric


class DeckFloorsClearedDojo(DeckLabelMetricDojo):
    """Deck -> stats.floorsCleared (FloorsClearedMetric)."""

    METRIC = FloorsClearedMetric


class DeckTotalCombatsDojo(DeckLabelMetricDojo):
    """Deck -> stats.totalCombats (TotalCombatsMetric)."""

    METRIC = TotalCombatsMetric


class WinDojo(MultiCardBinaryClassificationDojo):
    """Deck -> win (WinMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or WinMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(deck_box, WinMetric.LABEL_COLUMN),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class CharacterDojo(MultiCardFixedClassificationDojo):
    """Deck -> character (CharacterPredictionMetric).

    label_values=CharacterPredictionMetric.LABEL_VALUES unchanged - it
    already includes OTHER_LABEL (see that class's own docstring)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or CharacterPredictionMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                deck_box,
                CharacterPredictionMetric.LABEL_COLUMN,
                label_caster=str,
            ),
            label_values=CharacterPredictionMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
