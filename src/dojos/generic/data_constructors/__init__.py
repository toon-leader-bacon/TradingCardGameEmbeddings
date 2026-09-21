"""One DataConstructor (see data_constructor.py) per metric FAMILY - the
Template Method bases under src/data_refinement/metrics/. One module per
class (see plans/dojo_v2.md's directory layout), grouped here rather than
under any single generic/*/ dojo shape package: a DataConstructor maps to
a metric family, not to a (input shape, task shape) dojo cell, and several
of these are consumed by more than one shape package (DeckLabelDataConstructor)
or share private uuid-resolution helpers with constructors that feed a
different shape package entirely (see _uuid_resolution.py). Every class is
re-exported here so `from src.dojos.generic.data_constructors import
XDataConstructor` keeps working unchanged for every existing caller.
"""

from src.dojos.generic.data_constructors.attacker_blocker_combat_outcome import (
    AttackerBlockerCombatOutcomeDataConstructor,
)
from src.dojos.generic.data_constructors.card_average import CardAverageDataConstructor
from src.dojos.generic.data_constructors.card_character_prediction import (
    CardCharacterPredictionDataConstructor,
)
from src.dojos.generic.data_constructors.deck_card_mask import (
    DeckCardMaskDataConstructor,
)
from src.dojos.generic.data_constructors.deck_label import DeckLabelDataConstructor
from src.dojos.generic.data_constructors.masked_field import MaskedFieldDataConstructor
from src.dojos.generic.data_constructors.masked_field_regression import (
    MaskedFieldRegressionDataConstructor,
)
from src.dojos.generic.data_constructors.pack_to_pick_choice_set import (
    PackToPickChoiceSetDataConstructor,
)
from src.dojos.generic.data_constructors.pick_number_decay_curve import (
    PickNumberDecayCurveDataConstructor,
)
from src.dojos.generic.data_constructors.pool_conditioned_pick import (
    PoolConditionedPickDataConstructor,
)

__all__ = [
    "AttackerBlockerCombatOutcomeDataConstructor",
    "CardAverageDataConstructor",
    "CardCharacterPredictionDataConstructor",
    "DeckCardMaskDataConstructor",
    "DeckLabelDataConstructor",
    "MaskedFieldDataConstructor",
    "MaskedFieldRegressionDataConstructor",
    "PackToPickChoiceSetDataConstructor",
    "PickNumberDecayCurveDataConstructor",
    "PoolConditionedPickDataConstructor",
]
