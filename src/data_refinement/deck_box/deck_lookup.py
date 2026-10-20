"""The read-only view of DeckBox, for consumers that must never write.

DeckLookup is DeckBox's read surface (get_by_uuid, get_by_uuids, all_uuids,
all_decks, version_for, card_binder_version_for, uuids_ranked_randomly)
without its write surface (create, create_if_absent, update, replace,
delete, load, save, flush), named as a typing.Protocol so a signature
can say it never writes. DeckBox satisfies it structurally, so no
wrapper object is built. Mirrors CardLookup
(../card_binder/card_lookup.py), which does the same for CardBinder.

It matters more here than for CardBinder: a single-path DeckBox.load()
writes straight to the on-disk file, so a stray mutating call through a
reader's reference has no save() step that could catch it. Every dojo,
data constructor and DeckBoxDealer types its deck box as DeckLookup.

Like CardLookup, no runtime enforcement: a caller holding a DeckBox
typed as DeckLookup could still cast it and write. Only the type checker
guards this.
"""

from typing import Iterable, Protocol, Sequence
from uuid import UUID

from src.schema.card import GenericDeck
from src.schema.game_id import GameId


class DeckLookup(Protocol):
    """Read-only lookup and enumeration over a multi-game deck store."""

    def get_by_uuid(self, nocab_uuid: UUID) -> GenericDeck | None:
        """See DeckBox.get_by_uuid()."""
        ...

    def get_by_uuids(self, nocab_uuids: Sequence[UUID]) -> list[GenericDeck | None]:
        """See DeckBox.get_by_uuids()."""
        ...

    def all_uuids(self, source_game: GameId | None = None) -> Iterable[UUID]:
        """See DeckBox.all_uuids()."""
        ...

    def all_decks(self, source_game: GameId) -> Iterable[GenericDeck]:
        """See DeckBox.all_decks()."""
        ...

    def version_for(self, source_game: GameId) -> str:
        """See DeckBox.version_for()."""
        ...

    def card_binder_version_for(self, source_game: GameId) -> str | None:
        """See DeckBox.card_binder_version_for()."""
        ...

    def uuids_ranked_randomly(
        self, source_game: GameId, seed: int | None = None
    ) -> Iterable[tuple[UUID, int, int]]:
        """See DeckBox.uuids_ranked_randomly()."""
        ...
