"""The many-to-one external-identifier -> nocab_uuid translation table.

See src/data_refinement/card_binder/README.md for the full design
this implements.

AliasLedger holds no card content and has no opinion about which
source "wins" a naming collision — that's CardBinder.add()'s job.
Private to CardBinder (see card_binder.py) — nothing outside
card_binder/ ever holds an AliasLedger reference directly; every
interaction goes through CardBinder's own get_by_alias()/
register_alias() (a Facade, PATTERNS.md).
"""

import json
from pathlib import Path
from uuid import UUID

from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class AliasLedger:
    """In-memory (source_game, data_source, source_id) -> nocab_uuid table.

    source_game is part of the key (not just data_source) so two
    unrelated sources across different games can never collide even
    if their id-string spaces happen to overlap.
    """

    def __init__(self) -> None:
        """Construct an empty ledger.

        Inputs: none.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._uuid_by_alias: dict[tuple[GameId, DataSource, str], UUID] = {}

    def resolve(
        self, source_game: GameId, data_source: DataSource, source_id: str
    ) -> UUID | None:
        """Look up the nocab_uuid a given external identifier resolves to.

        Inputs:
            source_game: which game's namespace to look in.
            data_source: which external system minted source_id.
            source_id: that system's own id.
        Output: the resolved nocab_uuid, or None if this
            (source_game, data_source, source_id) triple isn't
            registered.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> ledger = AliasLedger()
            >>> ledger.resolve(GameId.MTG, DataSource.ARENA, "76497")
        """
        return self._uuid_by_alias.get((source_game, data_source, source_id))

    def register(
        self,
        source_game: GameId,
        data_source: DataSource,
        source_id: str,
        nocab_uuid: UUID,
    ) -> None:
        """Record that one external identifier resolves to nocab_uuid.

        Overwrites any prior mapping for this exact
        (source_game, data_source, source_id) triple — last write wins.

        Inputs:
            source_game: which game's namespace this identifier
                belongs to.
            data_source: which external system minted source_id.
            source_id: that system's own id.
            nocab_uuid: the nocab_uuid this identifier should resolve
                to.
        Output: none.
        Side effects: mutates this ledger's in-memory index.
        Exceptions: none.
        """
        self._uuid_by_alias[(source_game, data_source, source_id)] = nocab_uuid

    @staticmethod
    def load(path: Path, source_game: GameId) -> "AliasLedger":
        """Load one game's alias entries from path into a fresh ledger.

        Symmetric with save(), and useful for testing/inspecting an
        AliasLedger in isolation — NOT what CardBinder.load() uses
        internally: CardBinder.load() must remap a path's own
        nocab_uuids (as written by whatever save() produced that
        path's main file) to whatever card add() actually stored them
        as, which may differ on a cross-file collision — a
        registration this method has no way to perform, since it has
        no knowledge of any CardBinder. CardBinder.load() therefore
        reads a path's sibling alias-ledger file itself and calls
        register_alias() per (remapped) entry, rather than calling
        this method.

        source_game is required because path's rows (mirroring
        save()'s own on-disk format) don't carry it themselves — it's
        implied by which per-game file path is.

        Inputs:
            path: an alias-ledger file (e.g.
                data/final/cards/mtg.alias_ledger.jsonl), as written by
                save().
            source_game: which game every entry in path belongs to.
        Output: a fresh AliasLedger containing every entry in path.
        Side effects: reads path.
        Exceptions: raises if path doesn't exist or contains a line
            that doesn't deserialize to a valid entry.

        Example:
            >>> path = Path("data/final/cards/mtg.alias_ledger.jsonl")
            >>> ledger = AliasLedger.load(path, GameId.MTG)
        """
        ledger = AliasLedger()
        with open(path, "r", encoding="utf-8") as ledger_file:
            for line in ledger_file:
                row = json.loads(line)
                ledger.register(
                    source_game,
                    DataSource(row["data_source"]),
                    row["source_id"],
                    UUID(row["nocab_uuid"]),
                )
        return ledger

    def save(self, path: Path, source_game: GameId) -> None:
        """Write this ledger's entries for one game to path, as JSONL.

        Inputs:
            path: destination file (e.g.
                data/final/cards/<game>.alias_ledger.jsonl).
                Overwritten if it already exists.
            source_game: which game's subset of this ledger to write.
        Output: none.
        Side effects: creates path's parent directory if missing;
            writes/overwrites path.
        Exceptions: raises on failure to write path.

        Example:
            >>> ledger.save(Path("data/final/cards/mtg.alias_ledger.jsonl"), GameId.MTG)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as ledger_file:
            for (
                game,
                data_source,
                source_id,
            ), nocab_uuid in self._uuid_by_alias.items():
                if game != source_game:
                    continue
                row = {
                    "data_source": data_source.value,
                    "source_id": source_id,
                    "nocab_uuid": str(nocab_uuid),
                }
                ledger_file.write(json.dumps(row) + "\n")
