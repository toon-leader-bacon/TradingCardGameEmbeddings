"""Thin per-metric wrappers over DeckLabelMetric's int64-labeled
concrete subclasses (src/data_refinement/metrics/sts_gg/deck_label_metrics.py).

Each wrapper is a thin MultiCardRegressionDojo subclass - it adds no
behavior of its own, only configuration: it pulls its paired metric
class's own LABEL_COLUMN/DEFAULT_OUTPUT_PATH ClassVars by reference
(never duplicated as a literal) and passes them to
MultiCardRegressionDojo.__init__ via a DeckLabelDataConstructor
configured for that one label column. No wrapper instantiates its
paired metric class - that class's own constructor needs a card_binder/
deck_box a dojo has no reason to fabricate.

SCOPE: the 8 int64-labeled DeckLabelMetric subclasses are wrapped as
MultiCardRegressionDojo subclasses (RelicCountMetric,
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
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo


class DeckRelicCountDojo(MultiCardRegressionDojo):
    """Deck -> final relicCount (RelicCountMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or RelicCountMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, RelicCountMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class DeckTotalDamageTakenDojo(MultiCardRegressionDojo):
    """Deck -> stats.totalDamageTaken (TotalDamageTakenMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or TotalDamageTakenMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, TotalDamageTakenMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class DeckTotalCardsPickedDojo(MultiCardRegressionDojo):
    """Deck -> stats.totalCardsPicked (TotalCardsPickedMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or TotalCardsPickedMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, TotalCardsPickedMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class DeckTotalCardsSkippedDojo(MultiCardRegressionDojo):
    """Deck -> stats.totalCardsSkipped (TotalCardsSkippedMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or TotalCardsSkippedMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, TotalCardsSkippedMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class DeckTotalTurnsDojo(MultiCardRegressionDojo):
    """Deck -> stats.totalTurns (TotalTurnsMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or TotalTurnsMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, TotalTurnsMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class DeckElitesKilledDojo(MultiCardRegressionDojo):
    """Deck -> stats.elitesKilled (ElitesKilledMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or ElitesKilledMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, ElitesKilledMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class DeckFloorsClearedDojo(MultiCardRegressionDojo):
    """Deck -> stats.floorsCleared (FloorsClearedMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or FloorsClearedMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, FloorsClearedMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class DeckTotalCombatsDojo(MultiCardRegressionDojo):
    """Deck -> stats.totalCombats (TotalCombatsMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or TotalCombatsMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, TotalCombatsMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class WinDojo(MultiCardBinaryClassificationDojo):
    """Deck -> win (WinMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or WinMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, WinMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CharacterDojo(MultiCardFixedClassificationDojo):
    """Deck -> character (CharacterPredictionMetric).

    label_values=CharacterPredictionMetric.LABEL_VALUES unchanged - it
    already includes OTHER_LABEL (see that class's own docstring)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CharacterPredictionMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder,
                deck_box,
                CharacterPredictionMetric.LABEL_COLUMN,
                label_caster=str,
            ),
            label_values=CharacterPredictionMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
