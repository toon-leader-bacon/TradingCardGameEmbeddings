"""The read-only view of CardBinder, for consumers that must never write.

CardLookup is CardBinder's own read surface (get_by_uuid, get_by_name,
get_by_alias, get_by_name_regex, all_uuids) minus its write surface
(add, register_alias, load, save) — not a new capability, just that
subset named as its own typing.Protocol so a function's signature can
guarantee it never calls a write method, with zero runtime cost:
CardBinder already satisfies this Protocol structurally, so no wrapper
object is ever constructed. Any consumer that only needs to read
(training/'s Dojo contract today; data_refinement's own stages or
evaluation/ potentially later) should type against CardLookup, not
CardBinder, even when a real CardBinder is what gets passed in.

Deliberately does NOT provide runtime enforcement (a determined caller
holding a CardLookup-typed reference that's actually a CardBinder
could still call .add() by casting) — see plans/training_pipeline.md
for why a heavier wrapper object was considered and rejected as
unnecessary for this project's scale.
"""

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

    def get_by_name(self, source_game: GameId, name: str) -> GenericCard | None:
        """See CardBinder.get_by_name()."""
        ...

    def get_by_alias(
        self, source_game: GameId, data_source: DataSource, source_id: str
    ) -> GenericCard | None:
        """See CardBinder.get_by_alias()."""
        ...

    def get_by_name_regex(self, source_game: GameId, pattern: str) -> list[GenericCard]:
        """See CardBinder.get_by_name_regex()."""
        ...

    def all_uuids(self) -> Iterable[UUID]:
        """See CardBinder.all_uuids()."""
        ...
