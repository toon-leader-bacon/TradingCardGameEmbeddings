"""Thin wrapper over WinningDeckMaskedCardMetric
(src/data_refinement/metrics/isotropic/summary/deck_card_mask_metric.py),
the isotropic DeckCardMaskMetric: the winner's final deck with every copy
of one non-basic card removed -> that card's name.

Same shape as play_gwent's LeaderMaskedFromDeckDojo:
MultiCardFixedClassificationDojo fed by DeckCardMaskDataConstructor,
which drops every copy of target_card_uuid before the deck reaches the
encoder, so no masking Mod is needed.

label_values is the metric's own LABEL_VALUES (every non-basic Dominion
card name, ~807), although only the ~160 cards isotropic's era had ever
appear as labels.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.deck_card_mask_metric import (
    WinningDeckMaskedCardMetric,
)
from src.dojos.generic.data_constructors import DeckCardMaskDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.schema.holdout import HoldoutSpec


class WinningDeckMaskedCardDojo(MultiCardFixedClassificationDojo):
    """Winning deck, one card masked out -> that card's name
    (WinningDeckMaskedCardMetric)."""

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
        """
        Inputs:
            card_binder: card lookup for the Dominion nocab_uuids.
            holdout: card holdout shared by every dojo in a run.
            deck_box: the isotropic metrics' private DeckBox.
            card_embedding_size: width of the encoder's card embeddings.
            path_to_training_data: overrides the metric's
                DEFAULT_OUTPUT_PATH.
            name: Trainer-facing name and split-file prefix (see
                DojoConfig.name).
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file or deck box built against another CardBinder.
        Output: none (constructor).
        Side effects: see MultiCardFixedClassificationDojo (may write
            split files).
        Exceptions: see MultiCardFixedClassificationDojo.

        Example:
            >>> WinningDeckMaskedCardDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or WinningDeckMaskedCardMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckCardMaskDataConstructor(deck_box, "label"),
            label_values=WinningDeckMaskedCardMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            deck_box=deck_box,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
