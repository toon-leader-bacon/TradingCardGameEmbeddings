"""A CardLookup over several already-loaded per-game card stores.

CardShelf (src/training/dojo_catalog.py) holds one CardBinder per game. A
dojo that trains on cards from several games (the cross-game rarity dojo)
needs one CardLookup that sees them all. Loading every game's file into a
second merged CardBinder would hold each game's cards in memory twice
whenever the same run also has single-game dojos, so this Composite
forwards to the binders the shelf already holds.
"""

from itertools import chain
from typing import Iterable, Mapping
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class MultiGameCardLookup:
    """CardLookup over one CardLookup per game (Composite).

    Game-keyed calls go to that game's lookup; a call for a game it does
    not hold behaves as a miss (empty list, None, or an empty iterable),
    except version_for, which raises because a version of nothing would
    pass a stale-metric check silently.

    Inputs (constructor): lookups (game -> its CardLookup, at least one).
    Output: n/a.
    Side effects: none.
    Exceptions: ValueError from the constructor if lookups is empty.

    Example:
        >>> lookup = MultiGameCardLookup({GameId.GWENT: gwent, GameId.MTG: mtg})
        >>> lookup.get_by_uuid(some_gwent_uuid).source_game
        <GameId.GWENT: 'gwent'>
    """

    def __init__(self, lookups: Mapping[GameId, CardLookup]) -> None:
        if not lookups:
            raise ValueError("a MultiGameCardLookup needs at least one game")
        self._lookups = dict(lookups)

    def get_by_uuid(self, nocab_uuid: UUID) -> GenericCard | None:
        """See CardLookup.get_by_uuid(); the first game that holds it."""
        for lookup in self._lookups.values():
            card = lookup.get_by_uuid(nocab_uuid)
            if card is not None:
                return card
        return None

    def get_by_name(self, source_game: GameId, name: str) -> list[GenericCard]:
        """See CardLookup.get_by_name()."""
        lookup = self._lookups.get(source_game)
        return lookup.get_by_name(source_game, name) if lookup is not None else []

    def get_by_name_single(
        self, source_game: GameId, name: str, strict: bool = True
    ) -> GenericCard | None:
        """See CardLookup.get_by_name_single()."""
        lookup = self._lookups.get(source_game)
        return (
            lookup.get_by_name_single(source_game, name, strict)
            if lookup is not None
            else None
        )

    def get_by_alias(
        self, source_game: GameId, data_source: DataSource, source_id: str
    ) -> GenericCard | None:
        """See CardLookup.get_by_alias()."""
        lookup = self._lookups.get(source_game)
        return (
            lookup.get_by_alias(source_game, data_source, source_id)
            if lookup is not None
            else None
        )

    def get_by_name_regex(self, source_game: GameId, pattern: str) -> list[GenericCard]:
        """See CardLookup.get_by_name_regex()."""
        lookup = self._lookups.get(source_game)
        return (
            lookup.get_by_name_regex(source_game, pattern) if lookup is not None else []
        )

    def all_uuids(self, source_game: GameId | None = None) -> Iterable[UUID]:
        """See CardLookup.all_uuids(); None chains every game's, in the
        order the games were given."""
        if source_game is not None:
            lookup = self._lookups.get(source_game)
            return lookup.all_uuids(source_game) if lookup is not None else []
        return chain.from_iterable(
            lookup.all_uuids(game) for game, lookup in self._lookups.items()
        )

    def all_cards(self, source_game: GameId) -> Iterable[GenericCard]:
        """See CardLookup.all_cards()."""
        lookup = self._lookups.get(source_game)
        return lookup.all_cards(source_game) if lookup is not None else []

    def version_for(self, source_game: GameId) -> str:
        """See CardLookup.version_for(); raises ValueError for a game this
        lookup does not hold."""
        lookup = self._lookups.get(source_game)
        if lookup is None:
            raise ValueError(f"no card store for {source_game.value}")
        return lookup.version_for(source_game)
