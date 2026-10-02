"""Train-only mods that thin the card lists of a deck-shaped input.

A deck mod jitters which cards a multi-card or multi-group input shows,
so a deck-level head cannot lean on any one card. Every mod here:

- Thins only the groups it was built for. A multi-card input (one list
  of cards) is group 0; a multi-group input's groups are numbered in
  order. Other groups pass through as the same list objects. Which
  groups a dojo allows is data knowledge, kept in one table in
  src/training/dojo_catalog.py (DECK_MOD_GROUPS): an option-selection
  dojo's options group is never thinnable, since its label indexes it.
- Never empties a group: a non-empty group keeps at least one card, and
  an empty group passes through.
- Never mutates: kept cards are gathered into new lists (comprehensions
  and slicing only), in their original order; a datum with nothing
  removed comes back as the same object.
- Is best effort while applying, like the card-field mods: a datum it
  cannot thin (a single card, a group index the datum lacks, a
  non-card element) passes through unchanged and is counted in the
  ModTally. Constructor arguments are validated.

Opt-in only: a deck mod changes deck size and copy counts, so it must
never reach a dojo whose label depends on them (deck size, copy count,
held-out card). The catalog table is the gate.
"""

import logging
import random
from abc import abstractmethod
from typing import cast
from uuid import UUID

from src.dojos.mods.mod import Mod, ModTally
from src.schema.card import GenericCard
from src.schema.type_hints import TrainingDatum, TrainingInput

_logger = logging.getLogger(__name__)

# What a deck mod reads a datum's input as: its groups, in order
_Groups = list[list[GenericCard]]


class DeckThinningMod(Mod):
    """Template Method base: applies kept_cards() to each thinnable group
    of a datum, with the best-effort guard and the tally.

    Subclasses implement only kept_cards(cards). Always train-only: these
    are augmentations, never part of a task.

    tally: cards_seen counts the cards in thinnable groups, cards_changed
    the cards removed (so changed_fraction is the share of cards dropped),
    and cards_failed the data passed through unchanged after a failure.
    """

    def __init__(self, groups: frozenset[int], rng_seed: int | None = None) -> None:
        """Inputs: groups (indexes of the groups this mod may thin; a
            multi-card input is group 0), rng_seed (None: nondeterministic).
        Output: none. Side effects: none.
        Exceptions: ValueError if groups is empty or holds a non-int or
            negative index.
        """
        super().__init__(train_only=True)
        # Validate inputs: bool subclasses int, but is not an index
        if not groups:
            raise ValueError("groups must name at least one group")
        for index in groups:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(f"group index {index!r} must be an int >= 0")
        self.groups = groups
        self.rng = random.Random(rng_seed)
        self.tally: ModTally = ModTally()

    @abstractmethod
    def kept_cards(self, cards: list[GenericCard]) -> list[GenericCard]:
        """The cards this mod keeps from one non-empty group: a new list in
        the group's order, never empty; cards itself when it keeps all.

        Inputs: cards (non-empty). Output: list[GenericCard].
        Side effects: may advance self.rng. Exceptions: none expected.
        """
        ...

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        """A new datum with each thinnable group thinned; the label is
        unchanged, and data itself is returned when nothing was removed.

        Inputs: data (TrainingDatum: a multi-card or multi-group input).
        Output: TrainingDatum.
        Side effects: advances self.rng; updates self.tally; logs the
            first failure at warning level, later ones at debug.
        Exceptions: none for card data; a malformed datum (not an
            (input, label) pair) raises as unpacking would.

        Example:
            >>> CardDropoutMod(0.5, frozenset({0}), rng_seed=0).apply_single(([a, b, c], 1.0))
            ([a, c], 1.0)
        """
        training_input, label = data
        try:
            groups, is_multi_group = _groups_of(training_input, self.groups)
        except (TypeError, ValueError) as error:  # best effort, like CardFieldMod
            first = self.tally.record_failure(error)
            log = _logger.warning if first else _logger.debug
            log(
                "%s passed a datum through unchanged after: %r",
                type(self).__name__,
                error,
            )
            return data

        # Thin each allowed group into a new list; keep the others as they are
        thinned = [
            self._thinned_group(group) if index in self.groups else group
            for index, group in enumerate(groups)
        ]
        if all(new is old for new, old in zip(thinned, groups)):
            return data
        result = thinned if is_multi_group else thinned[0]
        return (cast(TrainingInput, result), label)

    def _thinned_group(self, cards: list[GenericCard]) -> list[GenericCard]:
        """kept_cards(cards) counted in the tally; an empty group passes.

        Inputs: cards. Output: list[GenericCard].
        Side effects: advances self.rng; updates self.tally.
        Exceptions: none expected.
        """
        if not cards:
            return cards
        kept = self.kept_cards(cards)
        self.tally.cards_seen += len(cards)
        self.tally.cards_changed += len(cards) - len(kept)
        return kept


class CardDropoutMod(DeckThinningMod):
    """Drop each card of a group with probability drop_probability; if
    every card would drop, keep one chosen uniformly instead."""

    def __init__(
        self,
        drop_probability: float,
        groups: frozenset[int] = frozenset({0}),
        rng_seed: int | None = None,
    ) -> None:
        """Inputs: drop_probability (in [0, 1]), groups, rng_seed (see
            DeckThinningMod). Output: none. Side effects: none.
        Exceptions: ValueError if drop_probability is outside [0, 1] or
            NaN; as DeckThinningMod for groups.
        """
        super().__init__(groups, rng_seed)
        # `not 0 <= p <= 1` also rejects NaN
        if not 0 <= drop_probability <= 1:
            raise ValueError(
                f"drop_probability must be in [0, 1], got {drop_probability}"
            )
        self.drop_probability = drop_probability

    def kept_cards(self, cards: list[GenericCard]) -> list[GenericCard]:
        """See DeckThinningMod.kept_cards: one roll per card.

        Example:
            >>> CardDropoutMod(1.0, rng_seed=0).kept_cards([a, b, c])
            [b]
        """
        kept = [card for card in cards if self.rng.random() >= self.drop_probability]
        if len(kept) == len(cards):
            return cards
        # Never empty: keep one card, chosen uniformly
        return kept or [self.rng.choice(cards)]


class CardSubsampleMod(DeckThinningMod):
    """Keep max(1, round(keep_fraction * n)) of a group's n cards, chosen
    uniformly without replacement, in their original order."""

    def __init__(
        self,
        keep_fraction: float,
        groups: frozenset[int] = frozenset({0}),
        rng_seed: int | None = None,
    ) -> None:
        """Inputs: keep_fraction (in (0, 1]), groups, rng_seed (see
            DeckThinningMod). Output: none. Side effects: none.
        Exceptions: ValueError if keep_fraction is outside (0, 1] or NaN;
            as DeckThinningMod for groups.
        """
        super().__init__(groups, rng_seed)
        # `not 0 < f <= 1` also rejects NaN
        if not 0 < keep_fraction <= 1:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        self.keep_fraction = keep_fraction

    def kept_cards(self, cards: list[GenericCard]) -> list[GenericCard]:
        """See DeckThinningMod.kept_cards: a fixed-size sample of slots.

        Example:
            >>> CardSubsampleMod(0.5, rng_seed=0).kept_cards([a, b, c, d])
            [a, d]
        """
        keep_count = max(1, round(self.keep_fraction * len(cards)))
        if keep_count >= len(cards):
            return cards
        # Sample slot indexes, then read them back in deck order
        slots = sorted(self.rng.sample(range(len(cards)), keep_count))
        return [cards[slot] for slot in slots]


class DuplicateCollapseMod(DeckThinningMod):
    """Keep one copy of each card (by nocab_uuid), at its first
    occurrence, so a head sees which cards a deck runs, not how many."""

    def __init__(
        self, groups: frozenset[int] = frozenset({0}), rng_seed: int | None = None
    ) -> None:
        """Inputs: groups, rng_seed (unused: the mod is deterministic,
            accepted so every deck spec builds alike). Output: none.
        Side effects: none. Exceptions: as DeckThinningMod for groups.
        """
        super().__init__(groups, rng_seed)

    def kept_cards(self, cards: list[GenericCard]) -> list[GenericCard]:
        """See DeckThinningMod.kept_cards: first copy of each card.

        Example:
            >>> DuplicateCollapseMod().kept_cards([copper, estate, copper])
            [copper, estate]
        """
        # A local index of first copies; cards itself is only read
        first_copies: dict[UUID, GenericCard] = {}
        for card in cards:
            first_copies.setdefault(card.nocab_uuid, card)
        kept = list(first_copies.values())
        return cards if len(kept) == len(cards) else kept


def _groups_of(
    training_input: TrainingInput, thinnable: frozenset[int]
) -> tuple[_Groups, bool]:
    """training_input read as groups: a multi-card input is one group
    (group 0), a multi-group input its own list of groups. The groups are
    training_input's own lists (read, never changed).

    Inputs: training_input, thinnable (the indexes the caller will thin).
    Output: (groups, is_multi_group).
    Side effects: none.
    Exceptions: TypeError for a single card or an element that is neither
        a card nor a list of cards; ValueError if a thinnable index is past
        the datum's last group.
    """
    if not isinstance(training_input, list):
        raise TypeError(
            f"a deck mod needs a list input, got {type(training_input).__name__}"
        )
    is_multi_group = any(isinstance(element, list) for element in training_input)
    groups = cast(_Groups, training_input if is_multi_group else [training_input])
    for group in groups:
        if not isinstance(group, list) or not all(
            isinstance(card, GenericCard) for card in group
        ):
            raise TypeError("a deck mod needs a list of cards or a list of card lists")
    missing = sorted(index for index in thinnable if index >= len(groups))
    if missing:
        raise ValueError(
            f"groups {missing} are past this datum's {len(groups)} group(s)"
        )
    return groups, is_multi_group
