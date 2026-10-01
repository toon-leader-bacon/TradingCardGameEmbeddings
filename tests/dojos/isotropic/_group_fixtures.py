"""A tiny Dominion world for the isotropic multi-group dojo tests: a
binder with a kingdom, the base supply, and a DeckBox of named groups."""

from dataclasses import dataclass
from uuid import UUID, uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.isotropic.pick_dojos import BASE_SUPPLY_NAMES
from src.schema.card import GenericCard, GenericDeck
from src.schema.game_id import GameId
from tests.dojos.isotropic._fixtures import dominion_card


@dataclass
class GroupWorld:
    binder: CardBinder
    box: DeckBox
    cards: dict[str, GenericCard]

    def add_group(self, names: list[str]) -> str:
        """Store a group of the named cards; return its uuid as a str."""
        deck = self.box.create(
            GenericDeck(
                nocab_uuid=uuid4(),
                source_game=GameId.DOMINION,
                name="group",
                card_nocab_uuids=[self.cards[name].nocab_uuid for name in names],
            )
        )
        return str(deck.nocab_uuid)

    def uuid(self, name: str) -> str:
        return str(self.cards[name].nocab_uuid)

    def uuid_of(self, name: str) -> UUID:
        return self.cards[name].nocab_uuid


KINGDOM_NAMES = ["Chapel", "Village", "Witch", "Moat"]


def group_world() -> GroupWorld:
    binder = CardBinder()
    cards = {
        name: binder.create(dominion_card(name))
        for name in [*KINGDOM_NAMES, *BASE_SUPPLY_NAMES]
    }
    return GroupWorld(binder, DeckBox(), cards)


def names(cards) -> list[str]:
    return [card.name for card in cards]


class HidingLookup:
    """A lookup that hides the given cards (as a split's holdout would)."""

    def __init__(self, inner: CardBinder, hidden: set[UUID]) -> None:
        self._inner = inner
        self._hidden = hidden

    def get_by_uuid(self, nocab_uuid: UUID) -> GenericCard | None:
        return (
            None if nocab_uuid in self._hidden else self._inner.get_by_uuid(nocab_uuid)
        )
