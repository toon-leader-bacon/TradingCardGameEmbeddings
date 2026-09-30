"""Best-effort augmentation mods that edit each card's raw_content.

Every mod here works on any TrainingInput shape (one card, a list of
cards, a list of card groups), edits each card independently, and follows
two rules:

- Never mutate: a changed card is a new GenericCard (GenericCardFactory or
  a new dict + dataclasses.replace); an unchanged card is passed through
  as the same object.
- Best effort while applying: a card the mod cannot edit (the field is
  missing, the data has an unexpected shape, anything else goes wrong) is
  passed through unchanged, and training continues. Constructor arguments
  ARE validated, so a bad configuration fails before any data is read.

Each mod keeps a ModTally of how many cards it saw and how many it
actually changed, so a mod that silently never fires (e.g. a key renamed
at ingestion) can be reported before a run.
"""

import dataclasses
import logging
import random
from abc import abstractmethod
from dataclasses import dataclass

from src.dojos.mods.mod import MASK_TOKEN, Mod, ModTally
from src.schema.card import GenericCard
from src.schema.card_factory import FieldPath, GenericCardFactory, MissingPathPolicy
from src.schema.type_hints import TrainingDatum, map_cards
from src.utils.drop_table import DropTable

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldMask:
    """One masking outcome: every path in paths is replaced by MASK_TOKEN.

    paths: FieldPaths (a top-level key is a one-step path). Empty means
        "mask nothing", the usual way to express "no mask this time" in a
        DropTable. No path may repeat or be a prefix of another (masking
        ("costs",) and then ("costs", "mana") would step into the mask
        token), so every FieldMask has one order-independent effect.
    """

    paths: tuple[FieldPath, ...] = ()

    def __post_init__(self) -> None:
        # Validate inputs: every path well formed on any card, none
        # repeating or nested inside another
        if not isinstance(self.paths, tuple):
            raise TypeError("paths must be a tuple of FieldPaths")
        for path in self.paths:
            # A bare str would pass check_path one character at a time
            if not isinstance(path, tuple):
                raise TypeError(f"path {path!r} must be a tuple (e.g. ('name',))")
            GenericCardFactory.check_path(path)
        for index, path in enumerate(self.paths):
            for other in self.paths[index + 1 :]:
                if _starts_with(path, other) or _starts_with(other, path):
                    raise ValueError(f"paths {path} and {other} overlap")

    @classmethod
    def of_keys(cls, *keys: str) -> "FieldMask":
        """A mask of top-level keys. Inputs: keys. Output: FieldMask.
        Side effects: none. Exceptions: as FieldMask validation.

        Example:
            >>> FieldMask.of_keys("faction", "faction-duo")
        """
        return cls(tuple((key,) for key in keys))

    def fits(self, card: GenericCard) -> bool:
        """Whether this mask means something for card: an empty mask
        ("nothing") always fits; otherwise the card must have at least one
        of its paths, and applied_to masks whichever it has. So
        of_keys("faction", "faction-duo") fits every Gwent card, masking
        faction-duo too on the few that have it.

        Inputs: card. Output: bool. Side effects: none.
        Exceptions: none (a path that runs into the wrong container type
            simply does not count as present).

        Example:
            >>> FieldMask.of_keys("faction", "faction-duo").fits(gwent_card)
            True
        """
        if not self.paths:
            return True
        return any(GenericCardFactory.has_field(card, path) for path in self.paths)

    def applied_to(self, card: GenericCard) -> GenericCard:
        """A new card with every path masked; paths the card lacks are
        skipped. card itself if nothing was masked.

        Inputs: card. Output: GenericCard. Side effects: none.
        Exceptions: TypeError from GenericCardFactory for a path that runs
            into the wrong container type (the mod's best-effort guard
            catches it).

        Example:
            >>> FieldMask((("name",),)).applied_to(card).raw_content["name"]
            '[MASK]'
        """
        result = card
        # Mask each path in turn; PASS leaves the card as is for a missing one
        for path in self.paths:
            result = GenericCardFactory.with_field(
                result, path, MASK_TOKEN, MissingPathPolicy.PASS
            )
        return result


class CardFieldMod(Mod):
    """Template Method base: applies modified_card() to every card of a
    datum, whatever its shape, with the best-effort guard and the tally.

    Subclasses implement only modified_card(card). Always train-only by
    default: these are augmentations, not part of any task.
    """

    def __init__(self, train_only: bool = True, rng_seed: int | None = None) -> None:
        """Inputs: train_only (see Mod), rng_seed (None: nondeterministic).
        Output: none. Side effects: none. Exceptions: none."""
        super().__init__(train_only=train_only)
        self.rng = random.Random(rng_seed)
        self.tally: ModTally = ModTally()

    @abstractmethod
    def modified_card(self, card: GenericCard) -> GenericCard:
        """The card with this mod applied, or card itself if unchanged.
        May raise on odd data; apply_single catches it.

        Inputs: card. Output: GenericCard. Side effects: may advance
        self.rng. Exceptions: any (caught by the caller).
        """
        ...

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        """A new datum with modified_card applied to each card (any input
        shape, via map_cards); the label is unchanged and data itself is
        never mutated.

        Inputs: data (TrainingDatum of any input shape).
        Output: TrainingDatum.
        Side effects: advances self.rng; updates self.tally.
        Exceptions: none for card data; a malformed datum (not a
            (input, label) pair) raises as unpacking would.

        Example:
            >>> ShuffleKeysMod(rng_seed=0).apply_single((card, 1.0))
            (GenericCard(...), 1.0)
        """
        training_input, label = data
        return (map_cards(training_input, self._guarded_modified_card), label)

    def _guarded_modified_card(self, card: GenericCard) -> GenericCard:
        """modified_card(card), or card unchanged if it raises; counts the
        card in self.tally. The first failure is logged at warning level
        (it is a bug or unexpected data), later ones at debug.

        Inputs: card. Output: GenericCard.
        Side effects: updates self.tally; may log. Exceptions: none.
        """
        self.tally.cards_seen += 1
        try:
            result = self.modified_card(card)
        except Exception as error:  # best effort: odd data must not stop training
            first = self.tally.record_failure(error)
            log = _logger.warning if first else _logger.debug
            log(
                "%s passed card %r through unchanged after: %r",
                type(self).__name__,
                card.name,
                error,
            )
            return card
        if result is not card:
            self.tally.cards_changed += 1
        return result


class ShuffleKeysMod(CardFieldMod):
    """Randomly reorder a card's top-level raw_content keys (values and
    nested content untouched), so a model cannot rely on field position.
    The serializer deliberately keeps insertion order for this mod."""

    def modified_card(self, card: GenericCard) -> GenericCard:
        """A new card with top-level keys in a random order; card itself
        when it has fewer than two keys or the draw keeps the same order.

        Inputs: card. Output: GenericCard. Side effects: advances self.rng.
        Exceptions: none expected.

        Example:
            >>> list(ShuffleKeysMod(rng_seed=1).modified_card(card).raw_content)
            ['power', 'name', 'faction']
        """
        keys = list(card.raw_content)
        order = self.rng.sample(keys, len(keys))
        if order == keys:
            return card
        # A new top-level dict; nested values are shared, never mutated
        reordered = {key: card.raw_content[key] for key in order}
        return dataclasses.replace(card, raw_content=reordered)


class RandomKeyMaskMod(CardFieldMod):
    """With the given probability, mask one top-level key chosen uniformly
    from the keys the card has."""

    def __init__(
        self,
        probability: float = 1.0,
        train_only: bool = True,
        rng_seed: int | None = None,
    ) -> None:
        """Inputs: probability (chance a card gets a mask at all, in
        [0, 1]), train_only, rng_seed. Output: none. Side effects: none.
        Exceptions: ValueError if probability is outside [0, 1] or NaN."""
        super().__init__(train_only=train_only, rng_seed=rng_seed)
        # `not 0 <= p <= 1` also rejects NaN
        if not 0 <= probability <= 1:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        self.probability = probability

    def modified_card(self, card: GenericCard) -> GenericCard:
        """card with one random top-level key masked, or card itself (no
        roll this time, or no keys).

        Inputs: card. Output: GenericCard. Side effects: advances self.rng.
        Exceptions: none expected.

        Example:
            >>> RandomKeyMaskMod(rng_seed=0).modified_card(card).raw_content
            {'name': '[MASK]', 'power': 3}
        """
        keys = list(card.raw_content)
        if not keys or self.rng.random() >= self.probability:
            return card
        return FieldMask.of_keys(self.rng.choice(keys)).applied_to(card)


class WeightedFieldMaskMod(CardFieldMod):
    """Pull a FieldMask from a DropTable and apply it.

    Per card, the table is first filtered to the masks that fit the card
    (FieldMask.fits: empty, or at least one path present), so weight meant
    for fields this card lacks is spread over masks that do something.
    A mask whose paths are only partly present masks the ones that are.
    A table with no fitting mask leaves the card unchanged. "Mask nothing"
    is an empty FieldMask row, so the table itself sets how often anything
    is masked.
    """

    def __init__(
        self,
        table: DropTable[FieldMask],
        train_only: bool = True,
        rng_seed: int | None = None,
    ) -> None:
        """Inputs: table (DropTable of FieldMask; weights validated by its
        own construction), train_only, rng_seed. Output: none.
        Side effects: none.
        Exceptions: TypeError if any outcome in table (at any depth) is not
            a FieldMask, so a badly built table fails here, not per card.
        """
        super().__init__(train_only=train_only, rng_seed=rng_seed)
        # Any outcome surviving this filter is one that is not a FieldMask
        if table.filtered(lambda outcome: not isinstance(outcome, FieldMask)):
            raise TypeError(
                "every outcome of a WeightedFieldMaskMod table must be a FieldMask"
            )
        self.table = table

    def modified_card(self, card: GenericCard) -> GenericCard:
        """card with a pulled FieldMask applied, or card itself.

        Inputs: card. Output: GenericCard. Side effects: advances self.rng.
        Exceptions: whatever FieldMask.applied_to raises (caught by the
            best-effort guard).

        Example:
            >>> table = DropTable.of([(1, FieldMask()), (1, FieldMask.of_keys("faction"))])
            >>> WeightedFieldMaskMod(table, rng_seed=0).modified_card(card)
        """
        # Keep only the masks that mean something for this card
        fitting = self.table.filtered(lambda mask: mask.fits(card))
        if fitting is None:
            return card

        # Pull one and apply it
        return fitting.pull(self.rng).applied_to(card)


def _starts_with(path: FieldPath, prefix: FieldPath) -> bool:
    """Whether prefix is path's first len(prefix) steps (equal paths too).

    Inputs: path, prefix. Output: bool. Side effects: none.
    Exceptions: none.
    """
    return path[: len(prefix)] == prefix
