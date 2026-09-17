"""Masking metric: a dominiontabs card's primary type.

See src/data_refinement/metrics/generic/masked_field_metric.py for the
shared scan() sequence this fixes MASKED_FIELD/LABEL_VALUES/
eligibility for, and ./BRAINSTORM.md item 2 for the full design.

types is a LIST on every dominiontabs card (e.g. ["Action", "Attack"],
["Action", "Duration"] - 94 distinct combos across the 819-card
corpus, far too sparse to classify as a single combined label), unlike
gwent_one's scalar "type" field. This metric predicts only the first
(primary) listed type - 19 distinct values, confirmed by direct
sampling of data/raw/dominiontabs/cards_db.json - so _label_for_card()
below reads card.raw_content directly rather than going through
_raw_field_value() (which walks MASKED_FIELD to a single leaf value,
not a list index).

Sanity-checked against the live corpus: the 19 raw primary-type values
are extremely imbalanced (Action alone is 486/819 = 59.3%; Curse,
"Start Deck", Trash, and Reaction each have exactly ONE card total).
LABEL_VALUES below folds the six smallest classes (Artifact: 5, State:
3, Curse/"Start Deck"/Trash/Reaction: 1 each - 11 cards combined, each
individually too small to split across train/val/test) into
OTHER_LABEL via _label_or_other(), leaving 13 real classes plus
OTHER_LABEL. This does not fix Action's dominance - that imbalance is
real and left for the eventual Dojo/loss function to handle (e.g.
class weighting), not this metric's job to paper over.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.generic.masked_field_metric import (
    OTHER_LABEL,
    MaskedFieldMetric,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId


class TypeMaskMetric(MaskedFieldMetric):
    """Card -> primary type (types[0]), masked, falling back to
    OTHER_LABEL for the 6 smallest raw classes (see module docstring).
    Every dominiontabs card is eligible - types is present and
    non-empty on every card."""

    SOURCE_GAME: ClassVar[GameId] = GameId.DOMINION
    MASKED_FIELD: ClassVar[list[str]] = ["types"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "Action",
        "Event",
        "Treasure",
        "Ally",
        "Landmark",
        "Way",
        "Project",
        "Victory",
        "Trait",
        "Prophecy",
        "Night",
        "Hex",
        "Boon",
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/dominiontabs/type_mask.parquet"
    )

    def _label_for_card(self, card: GenericCard) -> str:
        return self._label_or_other(card.raw_content["types"][0])
