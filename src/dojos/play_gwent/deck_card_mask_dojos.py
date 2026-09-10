"""Thin per-metric wrappers over DeckCardMaskMetric's concrete
subclasses (src/data_refinement/metrics/play_gwent/).

Each wrapper is a thin MultiCardFixedClassificationDojo subclass - it
adds no behavior of its own, only configuration: it pulls its paired
metric class's own LABEL_VALUES/DEFAULT_OUTPUT_PATH ClassVars by
reference (never duplicated as a literal) and passes them to
MultiCardFixedClassificationDojo.__init__ via a
DeckCardMaskDataConstructor. Same convention as
src/dojos/sts_gg/deck_label_dojos.py's CharacterDojo - no wrapper
instantiates its paired metric class, since that class's own
constructor needs a card_lookup/deck_box a dojo has no reason to
fabricate.

SCOPE: LeaderMaskedFromDeckMetric is this family's only concrete
metric today.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)


class LeaderMaskedFromDeckDojo(MultiCardFixedClassificationDojo):
    """Deck (leader masked out) -> leader identity (LeaderMaskedFromDeckMetric).

    label_values=LeaderMaskedFromDeckMetric.LABEL_VALUES unchanged - it
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
            or LeaderMaskedFromDeckMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckCardMaskDataConstructor(
                card_binder, deck_box, "label"
            ),
            label_values=LeaderMaskedFromDeckMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
