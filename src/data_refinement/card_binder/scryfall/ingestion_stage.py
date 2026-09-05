"""Translates a Scryfall oracle-cards dump into stored cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and plans/card_binder_v2.md
for the full design this implements. This class is handed a live
CardBinder and owns its own duplicate-detection and
collision-resolution directly against it, rather than returning
IngestedCandidate data for a driver (build.py) to interpret centrally.
All source-specific identity-extraction knowledge lives here — this is
the one place that knows which raw JSON fields on a Scryfall row are
card identifiers versus printing/artwork metadata.

Confirmed by sampling data/raw/scryfall/oracle-cards-20260820090157.jsonl
directly:
  - Identity-bearing (extracted as secondary aliases): arena_id,
    mtgo_id, mtgo_foil_id, multiverse_ids (a list — 0/1/2+ entries).
  - NOT identity (deliberately excluded): id (printing-specific, not
    oracle_id), set_id, card_back_id, illustration_id (artwork/
    printing metadata); tcgplayer_id/cardmarket_id (commerce product
    ids, no current consumer).
  - Not every row has every field — e.g. Arena-illegal cards have no
    arena_id.

Design, per plans/card_binder_v2.md:
  - DEDUPLICATION: this is entirely the get_by_alias() lookup at the
    top of _ingest_row() — a hit means "this row is a duplicate of an
    already-stored card," a miss means "this is a genuinely new card."
    There is no other duplicate-detection logic anywhere in this
    file; oracle_id, resolved via
    binder.get_by_alias(self.SOURCE_GAME, DataSource.SCRYFALL,
    oracle_id), is a perfect natural key, so no name-based or
    heuristic fallback is needed (contrast pokemon_tcg's stage, which
    needs one since it has no equivalent natural key).
  - COLLISION RESOLUTION: once a duplicate is found, the policy is
    merge_strategies.keep_longer_content — a richness comparison
    shared with any other stage that wants the same policy by name.
  - source_game is NOT a parameter anywhere in this class — a stage
    always ingests exactly one game, so it's a class constant
    (SOURCE_GAME) instead. Passing it as an argument would let a
    caller construct a nonsensical call like
    ScryfallCardIngestionStage().ingest(path, binder) for a
    different game's binder without any way to catch the mismatch;
    tying it to the class instead makes that a non-issue by
    construction.
  - Every row's own primary alias (oracle_id) and every secondary
    alias it carries are registered on EVERY branch — including a row
    that matched an existing card but changed nothing — per
    plans/card_binder_v2.md's "Open risks": a losing/no-op row's
    identifier must never become a dead end for get_by_alias().
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_MULTIVERSE_IDS_FIELD = "multiverse_ids"
_SINGLE_VALUE_ALIAS_FIELDS = {
    "arena_id": DataSource.ARENA,
    "mtgo_id": DataSource.MTGO,
    "mtgo_foil_id": DataSource.MTGO,
}


class ScryfallCardIngestionStage:
    """Translates a Scryfall oracle-cards .jsonl dump directly into binder.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse a Scryfall oracle-cards .jsonl file, creating/updating
        cards directly on binder as a side effect.

        One _ingest_row() call per line of raw_path — no additional
        logic beyond calling that and collecting the non-None results.
        No filtering by layout/type — every row becomes exactly one
        create() or one considered-for-update() call.

        Inputs:
            raw_path: path to a Scryfall oracle-cards .jsonl file (one
                JSON object per line, e.g.
                data/raw/scryfall/oracle-cards-<timestamp>.jsonl).
            binder: the CardBinder to create/update cards on and
                register aliases against, as a side effect. Expected
                to already be loaded (e.g. via CardBinder.load()) by
                the caller — this method never calls load()/save()
                itself. Every card this call touches is stored under
                self.SOURCE_GAME (GameId.MTG) — there is no way to
                call this method for a different game.
        Output: nocab_uuid of every row that caused a create() or an
            actual content-changing replace() this call. A row that
            matched an existing card but changed nothing (an
            exact-duplicate re-fetch) is NOT included — see
            _ingest_row()'s docstring.
        Side effects: reads raw_path; creates/updates cards and
            registers aliases directly on binder, once per line.
        Exceptions: raises if raw_path doesn't exist, isn't valid
            JSONL, or a line is missing "oracle_id" or "name" (see
            _ingest_row()).

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])
            >>> stage = ScryfallCardIngestionStage()
            >>> changed_uuids = stage.ingest(
            ...     Path("data/raw/scryfall/oracle-cards-20260820090157.jsonl"),
            ...     binder,
            ... )
        """
        changed_uuids = []
        with open(raw_path, "r", encoding="utf-8") as raw_file:
            for line in raw_file:
                row = json.loads(line)
                result = self._ingest_row(row, binder)
                if result is not None:
                    changed_uuids.append(result)
        return changed_uuids

    def _ingest_row(self, row: dict, binder: CardBinder) -> UUID | None:
        """Create-or-merge one Scryfall row directly against binder.

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
            binder: the CardBinder to read from and write to.
        Output: the stored/canonical nocab_uuid for this row IF this
            call caused a create() or an actual content-changing
            replace(); None if this row matched an existing card but
            changed nothing.
        Side effects: creates or updates exactly one card on binder;
            registers one or more aliases on binder.
        Exceptions: raises if row is missing "oracle_id" or "name".
        """
        oracle_id = row["oracle_id"]
        existing = binder.get_by_alias(self.SOURCE_GAME, DataSource.SCRYFALL, oracle_id)

        if existing is None:
            # New card: build a fresh GenericCard and create it.
            card = self._build_card(row, oracle_id)
            binder.create(card)
            stored_uuid = card.nocab_uuid
            changed = True
        else:
            # Existing card: build a throwaway candidate and merge it.
            candidate = self._build_card(row, oracle_id)
            merged = merge_strategies.keep_longer_content(existing, candidate)
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(self.SOURCE_GAME, DataSource.SCRYFALL, oracle_id, stored_uuid)
        for data_source, source_id in self._extract_aliases(row):
            binder.register_alias(self.SOURCE_GAME, data_source, source_id, stored_uuid)

        return stored_uuid if changed else None

    def _build_card(self, row: dict, oracle_id: str) -> GenericCard:
        """Build a fresh GenericCard for one Scryfall row.

        Private helper — single consumer is _ingest_row(), from both
        its create-path and its throwaway-candidate-for-merge path
        (see that method's docstring — the candidate's own nocab_uuid
        is never actually stored on the merge path, only ever
        existing's is).

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
            oracle_id: row["oracle_id"], passed in rather than
                re-read, since callers already have it at hand.
        Output: a new GenericCard with a freshly minted nocab_uuid.
        Side effects: none.
        Exceptions: raises if row is missing "name".
        """
        return GenericCard(
            nocab_uuid=uuid4(),
            source_game=self.SOURCE_GAME,
            name=row["name"],
            raw_content=row,
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id=oracle_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )

    def _extract_aliases(self, row: dict) -> list[tuple[DataSource, str]]:
        """Extract every known secondary identifier present on a Scryfall row.

        Private helper — single consumer is _ingest_row(). Checks a
        fixed allowlist (arena_id, mtgo_id, mtgo_foil_id, each entry
        of multiverse_ids — see this module's docstring for why these
        four and not others) and tolerates any of them being absent —
        this is NOT the same as _ingest_row()'s required oracle_id;
        every entry here is optional per row.

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
        Output: one (DataSource, source_id) tuple per identity-bearing
            field actually present on row, in no particular required
            order. Empty list if row has none of the allowlisted
            fields.
        Side effects: none.
        Exceptions: none.
        """
        aliases = []

        for field_name, data_source in _SINGLE_VALUE_ALIAS_FIELDS.items():
            if field_name in row:
                aliases.append((data_source, str(row[field_name])))

        for multiverse_id in row.get(_MULTIVERSE_IDS_FIELD, []):
            aliases.append((DataSource.GATHERER, str(multiverse_id)))

        return aliases
