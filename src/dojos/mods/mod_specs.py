"""ModSpec: a frozen description of one augmentation mod, built into a
fresh Mod on demand.

A Mod holds per-dojo state (its random.Random and its ModTally), so one
Mod object must never be shared between dojos. A ModSpec is the shareable,
immutable recipe: per-game defaults (src/dojos/augmentation_defaults.py)
and run-config overrides (src/training/run_config.py) are both tuples of
ModSpecs, and each dojo builds its own Mods from them with its own seed.

One ModSpec subclass per card-field mod (polymorphism, not a kind switch).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.dojos.mods.card_field_mods import (
    CardFieldMod,
    FieldMask,
    RandomKeyMaskMod,
    ShuffleKeysMod,
    WeightedFieldMaskMod,
)
from src.utils.drop_table import DropTable


class ModSpec(ABC):
    """A recipe for one train-only augmentation mod.

    Every spec validates on construction by building one throwaway mod, so
    a bad config fails when the spec is made (config parse time) and the
    rules live only in the mods' own constructors. Subclasses must be
    frozen dataclasses: that is what runs __post_init__ (the validation)
    and keeps a shared spec immutable.
    """

    def __post_init__(self) -> None:
        # Validate inputs through the mod's own constructor
        self.build(rng_seed=0)

    @abstractmethod
    def build(self, rng_seed: int | None) -> CardFieldMod:
        """A new mod with its own random state and an empty tally.

        Inputs: rng_seed (None: nondeterministic). Output: CardFieldMod.
        Side effects: none. Exceptions: as the mod's constructor.
        """
        ...


@dataclass(frozen=True)
class ShuffleKeysSpec(ModSpec):
    """Builds a ShuffleKeysMod.

    Example:
        >>> ShuffleKeysSpec().build(rng_seed=0)
    """

    def build(self, rng_seed: int | None) -> CardFieldMod:
        """See ModSpec.build."""
        return ShuffleKeysMod(rng_seed=rng_seed)


@dataclass(frozen=True)
class RandomKeyMaskSpec(ModSpec):
    """Builds a RandomKeyMaskMod.

    probability: chance a card gets one key masked, in [0, 1].

    Example:
        >>> RandomKeyMaskSpec(probability=0.1).build(rng_seed=0)
    """

    probability: float

    def build(self, rng_seed: int | None) -> CardFieldMod:
        """See ModSpec.build."""
        return RandomKeyMaskMod(probability=self.probability, rng_seed=rng_seed)


@dataclass(frozen=True)
class WeightedFieldMaskSpec(ModSpec):
    """Builds a WeightedFieldMaskMod.

    table: the DropTable of FieldMasks (immutable, so shared safely
        across every mod built from this spec).

    Example:
        >>> WeightedFieldMaskSpec(DropTable.of([(1, FieldMask.of_keys("name"))]))
    """

    table: DropTable[FieldMask]

    def build(self, rng_seed: int | None) -> CardFieldMod:
        """See ModSpec.build."""
        return WeightedFieldMaskMod(self.table, rng_seed=rng_seed)
