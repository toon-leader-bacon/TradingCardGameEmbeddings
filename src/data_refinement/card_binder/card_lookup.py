"""The read-only view of CardBinder, for consumers that must never write.

CardLookup is CardBinder's own read surface (get_by_uuid, get_by_name,
get_by_name_single, get_by_alias, get_by_name_regex, all_uuids,
all_cards) minus its write surface (create, update, replace, delete,
register_alias, load, save) — not a new capability, just that subset
named as its own typing.Protocol so a function's signature can
guarantee it never calls a write method, with zero runtime cost:
CardBinder already satisfies this Protocol structurally, so no wrapper
object is ever constructed. Any consumer that only needs to read
(training/'s Dojo contract today; data_refinement's own metric stages
or evaluation/ potentially later) should type against CardLookup, not
CardBinder, even when a real CardBinder is what gets passed in.

A CardIngestionStage (see ingestion.py) does NOT use this type — per
plans/card_binder_v2.md, a stage needs both read and write access, and
is typed to accept a full CardBinder directly rather than a narrower
read/write-only Protocol (a deliberate, accepted choice — see that
plan's "Open risks").

Deliberately does NOT provide runtime enforcement (a determined caller
holding a CardLookup-typed reference that's actually a CardBinder
could still call .create() by casting) — see plans/training_pipeline.md
for why a heavier wrapper object was considered and rejected as
unnecessary for this project's scale.
"""

import re
from typing import Iterable, Protocol
from uuid import UUID

from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class CardLookup(Protocol):
    """Read-only lookup + enumeration over a multi-game card store."""

    def get_by_uuid(self, nocab_uuid: UUID) -> GenericCard | None:
        """See CardBinder.get_by_uuid()."""
        ...

    def get_by_name(self, source_game: GameId, name: str) -> list[GenericCard]:
        """See CardBinder.get_by_name()."""
        ...

    def get_by_name_single(
        self, source_game: GameId, name: str, strict: bool = True
    ) -> GenericCard | None:
        """See CardBinder.get_by_name_single()."""
        ...

    def get_by_alias(
        self, source_game: GameId, data_source: DataSource, source_id: str
    ) -> GenericCard | None:
        """See CardBinder.get_by_alias()."""
        ...

    def get_by_name_regex(self, source_game: GameId, pattern: str) -> list[GenericCard]:
        """See CardBinder.get_by_name_regex()."""
        ...

    def all_uuids(self, source_game: GameId | None = None) -> Iterable[UUID]:
        """See CardBinder.all_uuids()."""
        ...

    def all_cards(self, source_game: GameId) -> Iterable[GenericCard]:
        """See CardBinder.all_cards()."""
        ...

    def version_for(self, source_game: GameId) -> str:
        """See CardBinder.version_for()."""
        ...


def uuid_for_name_or_front_face(
    card_lookup: CardLookup, source_game: GameId, name: str
) -> UUID | None:
    """The one card a source's bare card name refers to, allowing for a
    split/MDFC card being spelled by its front face only.

    Inputs:
        card_lookup: the card store to search (read-only).
        source_game: which game's names to search.
        name: a card name as a raw source spells it (e.g. "Bruce
            Banner" for the binder's "Bruce Banner // The Hulk").
    Output: the matching nocab_uuid if get_by_name() returns exactly one
        card; else if exactly one card's name matches
        f"^{re.escape(name)}( //.*)?$" (name as a front face); else
        None. Ambiguity (2+ matches) is never guessed at - it is
        treated as no match.
    Side effects: none (read-only queries).
    Exceptions: none expected.

    Example:
        >>> uuid_for_name_or_front_face(binder, GameId.MTG, "Bruce Banner")
        UUID('...')
    """
    exact_matches = card_lookup.get_by_name(source_game, name)
    if len(exact_matches) == 1:
        return exact_matches[0].nocab_uuid

    front_face_matches = card_lookup.get_by_name_regex(
        source_game, f"^{re.escape(name)}( //.*)?$"
    )
    if len(front_face_matches) == 1:
        return front_face_matches[0].nocab_uuid
    return None
