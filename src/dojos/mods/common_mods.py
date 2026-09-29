"""Stock mods. Every mod here follows the no-mutation rule: it builds new
cards and lists and never changes the datum it was given (see
src/dojos/README.md, mods/)."""

import random
from typing import List, cast

from src.dojos.mods.mod import Mod
from src.schema.card import GenericCard
from src.schema.card_factory import FieldPath, GenericCardFactory, MissingPathPolicy
from src.schema.type_hints import TrainingDatum

_MASK_TOKEN = "[MASK]"


class NoOpMod(Mod):
    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        return data

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        return data


class MaskTargetKeyMod(Mod):
    """Replace one raw_content field of a single-card datum with "[MASK]".

    key: a top-level raw_content key (str), or a FieldPath for a nested
        field, e.g. ("card_faces", 0, "oracle_text"). Stored unchanged
        as `self.key`.
    missing: what to do when a card lacks the field
        (GenericCardFactory.with_field's MissingPathPolicy). The default,
        ADD, adds it as "[MASK]", as the old in-place mod did, so every
        card in a dojo looks the same whether or not it had the field.
        PASS leaves such cards unmasked; STRICT raises.

    Inputs (constructor): key (str | FieldPath), train_only (bool),
        missing (MissingPathPolicy).
    Side effects: none; the input card is never mutated.
    """

    def __init__(
        self,
        key: str | FieldPath,
        train_only: bool = True,
        missing: MissingPathPolicy = MissingPathPolicy.ADD,
    ) -> None:
        super().__init__(train_only=train_only)
        self.key = key
        self.missing = missing

    @property
    def _path(self) -> FieldPath:
        """self.key as a FieldPath (a bare str key is a one-step path)."""
        return (self.key,) if isinstance(self.key, str) else self.key

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        """A new datum with the card's field masked; `data` is unchanged.

        Inputs: data, a single-card TrainingDatum (GenericCard, label).
        Output: TrainingDatum with a new GenericCard and the same label
            (the input card itself if the field is missing under PASS).
        Side effects: none.
        Exceptions: AssertionError if the input is not a single card;
            KeyError / IndexError / TypeError from GenericCardFactory.with_field.

        Example:
            >>> MaskTargetKeyMod("faction").apply_single((card, 3))
            (GenericCard(... raw_content={..., 'faction': '[MASK]'}), 3)
        """
        card, label = data
        assert isinstance(
            card, GenericCard
        ), "MaskTargetKeyMod expects a single-card TrainingDatum"
        masked = GenericCardFactory.with_field(
            card, self._path, _MASK_TOKEN, self.missing
        )
        return (masked, label)

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        """apply_single on each datum, in order; `data` is unchanged."""
        return [self.apply_single(datum) for datum in data]


class ShuffleDeckMod(Mod):
    """Reorder the cards of a multi-card datum.

    Returns a new list in random order (random.sample from the global
    `random` module, as before) and never shuffles the caller's list in
    place. The card objects themselves are shared, not copied.

    Side effects: advances the global `random` state; `data` is unchanged.
    """

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        """A new datum with the same cards in a random order.

        Inputs: data, a multi-card TrainingDatum (list[GenericCard], label).
        Output: TrainingDatum with a new list and the same label.
        Side effects: advances the global `random` state.
        Exceptions: AssertionError if the input is not a list of cards.

        Example:
            >>> deck, label = ShuffleDeckMod().apply_single(([a, b, c], 1))
            >>> sorted(deck, key=id) == sorted([a, b, c], key=id)
            True
        """
        deck, label = data
        assert isinstance(deck, list) and all(
            isinstance(card, GenericCard) for card in deck
        ), "ShuffleDeckMod expects a multi-card TrainingDatum"
        cards = cast(List[GenericCard], deck)  # checked element-wise just above
        return (random.sample(cards, len(cards)), label)

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        """apply_single on each datum, in order; `data` is unchanged."""
        return [self.apply_single(datum) for datum in data]
