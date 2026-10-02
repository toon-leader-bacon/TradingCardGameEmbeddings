"""Thin wrapper over HeroMaskedFromDeckMetric
(src/data_refinement/metrics/fabtcg_decklists/), the FaB twin of
../play_gwent/deck_card_mask_dojos.py's LeaderMaskedFromDeckDojo. Its
rows point into the published FaB deck box, which the catalog passes in
(read-only).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.fabtcg_decklists.hero_masked_from_deck_metric import (
    HeroMaskedFromDeckMetric,
)
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.schema.holdout import HoldoutSpec


class HeroMaskedFromDeckDojo(MultiCardFixedClassificationDojo):
    """FaB deck (hero masked out) -> hero identity, over HERO_NAMES plus
    OTHER (HeroMaskedFromDeckMetric.LABEL_VALUES)."""

    METRIC = HeroMaskedFromDeckMetric

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """See LeaderMaskedFromDeckDojo; deck_box is the published FaB box."""
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckCardMaskDataConstructor(deck_box, "label"),
            label_values=self.METRIC.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
