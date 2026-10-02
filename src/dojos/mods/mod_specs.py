"""ModSpec: a frozen description of one augmentation mod, built into a
fresh Mod on demand.

A Mod holds per-dojo state (its random.Random and its ModTally), so one
Mod object must never be shared between dojos. A ModSpec is the shareable,
immutable recipe: per-game defaults (src/dojos/augmentation_defaults.py)
and run-config overrides (src/training/run_config.py) are both tuples of
ModSpecs, and each dojo builds its own Mods from them with its own seed.

One ModSpec subclass per mod (polymorphism, not a kind switch). The
card-field specs suit any dojo; the deck specs (DeckModSpec) thin card
lists and are opt-in per dojo, for the groups the catalog allows
(src/training/dojo_catalog.py, DECK_MOD_GROUPS).
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
from src.dojos.mods.deck_mods import (
    CardDropoutMod,
    CardSubsampleMod,
    DeckThinningMod,
    DuplicateCollapseMod,
)
from src.dojos.mods.mod import Mod
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
    def build(self, rng_seed: int | None) -> Mod:
        """A new mod with its own random state and an empty tally.

        Inputs: rng_seed (None: nondeterministic). Output: Mod.
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


class DeckModSpec(ModSpec):
    """A recipe for one deck-thinning mod (src/dojos/mods/deck_mods.py).

    groups: the indexes of the groups the mod may thin (a multi-card
    input is group 0). The catalog accepts a deck spec only for a dojo
    that allows every one of these groups, and appends its mod after the
    dojo's own task mods. Subclasses are frozen dataclasses with a groups
    field (default (0,)).
    """

    groups: tuple[int, ...]

    @abstractmethod
    def build(self, rng_seed: int | None) -> DeckThinningMod:
        """See ModSpec.build."""
        ...


@dataclass(frozen=True)
class CardDropoutSpec(DeckModSpec):
    """Builds a CardDropoutMod.

    drop_probability: chance each card of a thinned group is dropped, in
    [0, 1]. groups: see DeckModSpec.

    Example:
        >>> CardDropoutSpec(drop_probability=0.1).build(rng_seed=0)
    """

    drop_probability: float
    groups: tuple[int, ...] = (0,)

    def build(self, rng_seed: int | None) -> DeckThinningMod:
        """See ModSpec.build."""
        return CardDropoutMod(
            self.drop_probability, frozenset(self.groups), rng_seed=rng_seed
        )


@dataclass(frozen=True)
class CardSubsampleSpec(DeckModSpec):
    """Builds a CardSubsampleMod.

    keep_fraction: share of a thinned group's cards kept, in (0, 1].
    groups: see DeckModSpec.

    Example:
        >>> CardSubsampleSpec(keep_fraction=0.8, groups=(1,)).build(rng_seed=0)
    """

    keep_fraction: float
    groups: tuple[int, ...] = (0,)

    def build(self, rng_seed: int | None) -> DeckThinningMod:
        """See ModSpec.build."""
        return CardSubsampleMod(
            self.keep_fraction, frozenset(self.groups), rng_seed=rng_seed
        )


@dataclass(frozen=True)
class DuplicateCollapseSpec(DeckModSpec):
    """Builds a DuplicateCollapseMod. groups: see DeckModSpec.

    Example:
        >>> DuplicateCollapseSpec().build(rng_seed=0)
    """

    groups: tuple[int, ...] = (0,)

    def build(self, rng_seed: int | None) -> DeckThinningMod:
        """See ModSpec.build."""
        return DuplicateCollapseMod(frozenset(self.groups), rng_seed=rng_seed)
