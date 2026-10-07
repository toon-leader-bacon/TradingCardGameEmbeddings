from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from src.schema.type_hints import TrainingDatum

# What every masking mod writes in place of a hidden value
MASK_TOKEN = "[MASK]"


@dataclass
class ModTally:
    """How often a mod actually changed a card.

    cards_seen: cards the mod was applied to. Of those, cards_changed came
    back as a different card and cards_failed raised inside the mod (and
    were passed through unchanged); the rest were legitimate no-ops. A
    failure is a bug or unexpected data, so it is counted apart from a
    no-op. first_failure: repr of the first exception (None exactly while
    cards_failed is 0; record_failure keeps them in step).
    Mutable counters owned by one mod.
    """

    cards_seen: int = 0
    cards_changed: int = 0
    cards_failed: int = 0
    first_failure: str | None = None

    @property
    def changed_fraction(self) -> float:
        """cards_changed / cards_seen; 0.0 before any card is seen.
        Side effects: none. Exceptions: none."""
        return self.cards_changed / self.cards_seen if self.cards_seen else 0.0

    def record_failure(self, error: Exception) -> bool:
        """Count one failed card, keeping first_failure in step with
        cards_failed.

        Inputs: error. Output: True if this was the first failure.
        Side effects: updates the counters. Exceptions: none.

        Example:
            >>> ModTally().record_failure(KeyError("x"))
            True
        """
        self.cards_failed += 1
        if self.first_failure is not None:
            return False
        self.first_failure = repr(error)
        return True


class Mod(ABC):
    """A transformation applied to one or more TrainingDatum before training.

    train_only decides which splits this mod runs against. Defaults to
    True - the original use case (noise/augmentation a model shouldn't
    see at eval time: shuffling a deck, reordering json keys, etc).
    Pass train_only=False at construction for a mod that's a structural
    requirement of the task itself rather than augmentation - e.g. a
    masking mod, where the masked field must stay masked at test/
    validation time too, or the model can just read the answer off the
    input.

    Every mod must leave its input unchanged and return new objects; see
    src/dojos/README.md (mods/) for the rule and the helpers to use.

    tally: a ModTally for mods that count what they change (the card-field
    augmentations), None for mods that do not; ModPipeline.mod_tallies
    reports the non-None ones.
    """

    def __init__(self, train_only: bool = True) -> None:
        self.train_only = train_only
        self.tally: ModTally | None = None

    @abstractmethod
    def apply_single(self, data: TrainingDatum) -> TrainingDatum: ...

    def child_tallies(self) -> dict[str, ModTally]:
        """Tallies of mods this mod holds inside itself (a PerGameMod's
        per-game mods), keyed so ModPipeline.mod_tallies can report them
        under this mod's own key. Default: none.

        Inputs: none. Output: dict[str, ModTally] (the live objects).
        Side effects: none. Exceptions: none.
        """
        return {}

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        """apply_single on each datum, in order; data is unchanged.
        Override only for a mod that works across data (e.g. mixing two
        datums). Side effects: as apply_single. Exceptions: as apply_single."""
        return [self.apply_single(datum) for datum in data]
