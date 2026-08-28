"""Translates a Scryfall oracle-cards dump into candidate cards + aliases.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and this container's
README for the raw source this reads
(data/raw/scryfall/*.jsonl, written by
src/data_retrieval/scryfall/downloader.py).

All source-specific identity-extraction knowledge lives here, not in
CardBinder/AliasLedger (see ../README.md) — this is the one place that
knows which raw JSON fields on a Scryfall row are card identifiers
versus printing/artwork metadata, and that presence of any of them
(besides oracle_id) is per-row-optional.

Confirmed by sampling data/raw/scryfall/oracle-cards-20260820090157.jsonl
directly:
  - Identity-bearing (extracted as aliases): arena_id, mtgo_id,
    mtgo_foil_id, multiverse_ids (a list — 0/1/2+ entries).
  - NOT identity (deliberately excluded): id (printing-specific, not
    oracle_id), set_id, card_back_id, illustration_id (artwork/
    printing metadata); tcgplayer_id/cardmarket_id (commerce product
    ids, no current consumer).
  - Not every row has every field — e.g. Arena-illegal cards have no
    arena_id.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.ingestion import ExternalIdentifier, IngestedCandidate
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
    """Translates a Scryfall oracle-cards .jsonl dump into IngestedCandidates.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly (they go through
    CardIngestionStage/build_or_update_card_binder instead).
    """

    def ingest(self, raw_path: Path, source_game: GameId) -> list[IngestedCandidate]:
        """Parse a Scryfall oracle-cards .jsonl file into IngestedCandidates.

        One IngestedCandidate per line of raw_path, via
        _parse_row(). No filtering by layout/type, no deduplication by
        name — see
        src/data_refinement/card_binder/ingestion.py's module
        docstring.

        Inputs:
            raw_path: path to a Scryfall oracle-cards .jsonl file (one
                JSON object per line, e.g.
                data/raw/scryfall/oracle-cards-<timestamp>.jsonl).
            source_game: which game these cards belong to (expected to
                be GameId.MTG for this implementation, but not
                enforced).
        Output: one IngestedCandidate per line in raw_path.
        Side effects: none — reads raw_path, no other I/O.
        Exceptions: raises if raw_path doesn't exist, isn't valid
            JSONL, or a line is missing "oracle_id" or "name".

        Example:
            >>> stage = ScryfallCardIngestionStage()
            >>> candidates = stage.ingest(
            ...     Path("data/raw/scryfall/oracle-cards-20260820090157.jsonl"),
            ...     GameId.MTG,
            ... )
        """
        candidates = []
        with open(raw_path, "r", encoding="utf-8") as raw_file:
            for line in raw_file:
                row = json.loads(line)
                candidates.append(self._parse_row(row, source_game))
        return candidates

    def _parse_row(self, row: dict, source_game: GameId) -> IngestedCandidate:
        """Translate one Scryfall JSON row into an IngestedCandidate.

        Private helper — single consumer is ingest(). Composed of:
        building a GenericCard (nocab_uuid=uuid4(), source_game, name=
        row["name"], raw_content=row, provenance=Provenance(
        data_source=DataSource.SCRYFALL, source_id=row["oracle_id"],
        fetched_at=<now, or a stage-level constant>)) plus
        _extract_aliases(row) for its aliases list.

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
            source_game: which game this row belongs to.
        Output: an IngestedCandidate wrapping the translated
            GenericCard and every alias _extract_aliases() found.
        Side effects: none.
        Exceptions: raises if row is missing "oracle_id" or "name".
        """
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=source_game,
            name=row["name"],
            raw_content=row,
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id=row["oracle_id"],
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        return IngestedCandidate(card=card, aliases=self._extract_aliases(row))

    def _extract_aliases(self, row: dict) -> list[ExternalIdentifier]:
        """Extract every known secondary identifier present on a Scryfall row.

        Private helper — single consumer is _parse_row(). Checks a
        fixed allowlist (arena_id, mtgo_id, mtgo_foil_id, each entry
        of multiverse_ids — see this module's docstring for why these
        four and not others) and tolerates any of them being absent —
        this is NOT the same as _parse_row()'s required oracle_id;
        every entry here is optional per row.

        Inputs:
            row: one parsed JSON object from a Scryfall oracle-cards
                line.
        Output: one ExternalIdentifier per identity-bearing field
            actually present on row, in no particular required order.
            Empty list if row has none of the allowlisted fields.
        Side effects: none.
        Exceptions: none.
        """
        aliases = []

        for field_name, data_source in _SINGLE_VALUE_ALIAS_FIELDS.items():
            if field_name in row:
                aliases.append(ExternalIdentifier(data_source, str(row[field_name])))

        for multiverse_id in row.get(_MULTIVERSE_IDS_FIELD, []):
            aliases.append(ExternalIdentifier(DataSource.GATHERER, str(multiverse_id)))

        return aliases
