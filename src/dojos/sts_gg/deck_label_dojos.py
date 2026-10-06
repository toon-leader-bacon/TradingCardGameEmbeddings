"""Thin per-metric wrappers over every concrete DeckLabelMetric
(src/data_refinement/metrics/sts_gg/deck_label_metrics.py and
ascension_prediction_metric.py).

SCOPE: the 9 int64-labeled DeckLabelMetric subclasses are wrapped as
DeckLabelMetricDojo subclasses (../generic/paired_metric_dojos.py)
(RelicCountMetric, TotalDamageTakenMetric, TotalCardsPickedMetric,
TotalCardsSkippedMetric, TotalTurnsMetric, ElitesKilledMetric,
FloorsClearedMetric, TotalCombatsMetric, AscensionPredictionMetric -
ascension as a regression, since its levels are ordered); WinMetric
(bool) is wrapped as a DeckBinaryLabelMetricDojo subclass (WinDojo);
CharacterPredictionMetric (str, closed vocabulary) is wrapped as a
DeckFixedLabelMetricDojo subclass (CharacterDojo). Each of these three
base classes only names METRIC - no wrapper here hand-rolls a
constructor. KilledByMetric (str, closed vocabulary, null on a win) is
wrapped like CharacterPredictionMetric (KilledByDojo);
DeckLabelDataConstructor skips its null rows.

sts_gg lists winning runs only, so WinDojo and KilledByDojo have no
signal over sts_gg's own outputs (see deck_label_metrics.py's WINS
ONLY note) and have no sts_gg catalog key; the sts2_runs catalog keys
reuse every wrapper here over
../../data_refinement/metrics/sts2_runs/'s outputs, which share each
sts_gg metric's LABEL_COLUMN (and LABEL_VALUES).

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

from src.data_refinement.metrics.sts_gg.ascension_prediction_metric import (
    AscensionPredictionMetric,
)
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
    ElitesKilledMetric,
    FloorsClearedMetric,
    KilledByMetric,
    RelicCountMetric,
    TotalCardsPickedMetric,
    TotalCardsSkippedMetric,
    TotalCombatsMetric,
    TotalDamageTakenMetric,
    TotalTurnsMetric,
    WinMetric,
)
from src.dojos.generic.paired_metric_dojos import (
    DeckBinaryLabelMetricDojo,
    DeckFixedLabelMetricDojo,
    DeckLabelMetricDojo,
)


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


class DeckAscensionDojo(DeckLabelMetricDojo):
    """Deck -> ascension level (AscensionPredictionMetric)."""

    METRIC = AscensionPredictionMetric


class WinDojo(DeckBinaryLabelMetricDojo):
    """Deck -> win (WinMetric)."""

    METRIC = WinMetric


class CharacterDojo(DeckFixedLabelMetricDojo):
    """Deck -> character (CharacterPredictionMetric).

    label_values=CharacterPredictionMetric.LABEL_VALUES unchanged - it
    already includes OTHER_LABEL (see that class's own docstring)."""

    METRIC = CharacterPredictionMetric


class KilledByDojo(DeckFixedLabelMetricDojo):
    """Deck -> the encounter that ended a lost run (KilledByMetric).

    label_values=KilledByMetric.LABEL_VALUES unchanged - it already
    includes OTHER_LABEL. A won run's null label has no row to learn
    from (DeckLabelDataConstructor skips it)."""

    METRIC = KilledByMetric
