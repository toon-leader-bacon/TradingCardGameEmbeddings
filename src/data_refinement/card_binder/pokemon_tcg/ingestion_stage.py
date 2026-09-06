"""Translates pokemon-tcg-data per-set card dumps into stored cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and plans/card_binder_v2.md
for the design this implements. This class is handed a live CardBinder
and owns its own duplicate-detection and collision-resolution directly
against it — see src/data_refinement/card_binder/scryfall/ingestion_stage.py
for the reference shape this follows.

This stage reads a *directory* of .json files — one per set, each a
JSON array of card objects — since pokemon-tcg-data ships that way.

Confirmed by sampling data/raw/pokemon_tcg/cards/*.json directly
(20,444 rows across every set file currently downloaded):
  - "id" (e.g. "swsh8-1") is printing-specific (set code + card
    number) — used as Provenance.source_id, but it is NOT a usable
    identity key on its own: every printing already has a distinct
    id, so keying off it directly would never collapse anything.
  - "set" — the field a first-pass design assumed would carry set
    metadata — is `None` on every single sampled row (confirmed
    across all 20,444). It cannot be used.
  - The set code IS reliably recoverable from "id" itself:
    id.rsplit("-", 1)[0] (e.g. "swsh8-1" -> "swsh8") holds for every
    sampled row (zero exceptions).
  - IDENTITY: (name, set code) is the identity key — confirmed against
    real duplicate-name cases within a single set file (e.g. "Mew V"
    appears 3 times in swsh8.json, at ids swsh8-113/250/251): these
    are alternate-art/rarity reprints with IDENTICAL rules text
    (attacks, hp, types all equal) — genuinely the same card for this
    project's purposes, same relationship as Scryfall's multiple
    printings of one oracle_id. Unlike spire-codex's Strike/Defend
    (which are near-identical but genuinely DIFFERENT cards per
    character), same-name-same-set Pokemon rows checked so far are
    the same card. Cross-set same-name cards (e.g. "Pikachu" in
    different sets/eras) are treated as distinct cards, per this
    project's own stated intent — a different set code means a
    different identity, never merged.
  - Since (name, set code) has no CardBinder-native lookup, identity
    resolution is: binder.get_by_name(self.SOURCE_GAME, name) (a
    list), then filter that list by comparing each match's own set
    code (re-derived from its stored raw_content["id"]) against this
    row's set code.
  - No secondary identifier system exists in this data comparable to
    Scryfall's arena_id/mtgo_id/multiverse_ids — no extra aliases are
    ever registered for this source, only each row's own id.
    "nationalPokedexNumbers" identifies a Pokemon *species*, not a
    printing — deliberately NOT extracted as an alias: many distinct
    printings share the same dex number, and an alias key has no
    defined multi-owner behavior.

Collision policy is merge_strategies.keep_longer_content, same as
ScryfallCardIngestionStage.
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


class PokemonTcgCardIngestionStage:
    """Translates a pokemon-tcg-data cards/ directory directly into binder.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly.
    """

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse every *.json file under raw_path, creating/updating
        cards directly on binder as a side effect.

        One _ingest_row() call per card object across every *.json
        file directly under raw_path (each file is a JSON array) — no
        additional logic beyond calling that and collecting the
        non-None results.

        Inputs:
            raw_path: path to a directory of pokemon-tcg-data per-set
                .json files (e.g. data/raw/pokemon_tcg/cards, written
                by src/data_retrieval/pokemon_tcg/downloader.py). Not
                a single file — see this module's docstring.
            binder: the CardBinder to create/update cards on and
                register aliases against, as a side effect. Every card
                this call touches is stored under self.SOURCE_GAME
                (GameId.POKEMON).
        Output: nocab_uuid of every row that caused a create() or an
            actual content-changing replace() this call. A row that
            matched an existing card but changed nothing (an
            exact-duplicate re-fetch, or an alternate-art reprint with
            identical rules text — see this module's docstring) is NOT
            included.
        Side effects: reads raw_path's *.json files; creates/updates
            cards and registers aliases directly on binder.
        Exceptions: raises if raw_path doesn't exist, isn't a
            directory, contains no *.json files, or a file isn't a
            JSON array of objects each carrying "id" and "name".

        Example:
            >>> binder = CardBinder.load([Path("data/final/cards/pokemon.jsonl")])
            >>> stage = PokemonTcgCardIngestionStage()
            >>> changed_uuids = stage.ingest(
            ...     Path("data/raw/pokemon_tcg/cards"), binder
            ... )
        """
        if not raw_path.is_dir():
            raise ValueError(f"pokemon_tcg ingest: {raw_path} is not a directory")

        set_paths = sorted(raw_path.glob("*.json"))
        if not set_paths:
            raise ValueError(
                f"pokemon_tcg ingest: no *.json files found under {raw_path}"
            )

        changed_uuids = []
        for set_path in set_paths:
            with open(set_path, "r", encoding="utf-8") as set_file:
                rows = json.load(set_file)
            for row in rows:
                result = self._ingest_row(row, binder)
                if result is not None:
                    changed_uuids.append(result)
        return changed_uuids

    def _ingest_row(self, row: dict, binder: CardBinder) -> UUID | None:
        """Create-or-merge one pokemon-tcg-data row directly against binder.

        Private helper — single consumer is ingest(). THIS METHOD IS
        WHERE DEDUPLICATION HAPPENS: unlike ScryfallCardIngestionStage,
        this source has no perfect natural key, so the identity check
        is a heuristic — binder.get_by_name(self.SOURCE_GAME,
        row["name"]) followed by filtering that list to the one (if
        any) whose own set code (_set_code() of ITS stored
        raw_content["id"]) equals this row's own set code. A match
        means row is a duplicate of that card; no match means row is
        new — see this module's docstring for why (name, set code) is
        the right identity key for this data.

        If no existing card resolves: builds a fresh GenericCard and
        calls binder.create().

        If an existing card resolves: builds the same kind of
        GenericCard as a throwaway candidate, then calls
        merge_strategies.keep_longer_content(existing, candidate).
        Compares the result to existing BY VALUE (dataclass equality):
        if different, calls binder.replace(existing.nocab_uuid,
        merged).

        Regardless of branch: registers row's own id alias against
        whichever uuid ended up stored — on every branch, not only the
        content-changing ones.

        Inputs:
            row: one parsed JSON object from a pokemon-tcg-data
                cards/*.json file's array.
            binder: the CardBinder to read from and write to.
        Output: the stored/canonical nocab_uuid for this row IF this
            call caused a create() or an actual content-changing
            replace(); None if this row matched an existing card but
            changed nothing.
        Side effects: creates or updates exactly one card on binder;
            registers one alias on binder.
        Exceptions: raises if row is missing "id" or "name".
        """
        printing_id = row["id"]
        set_code = self._set_code(printing_id)
        existing = self._find_matching_printing(row["name"], set_code, binder)

        if existing is None:
            card = self._build_card(row, printing_id)
            binder.create(card)
            stored_uuid = card.nocab_uuid
            changed = True
        else:
            candidate = self._build_card(row, printing_id)
            merged = merge_strategies.keep_longer_content(existing, candidate)
            changed = merged != existing
            if changed:
                binder.replace(existing.nocab_uuid, merged)
            stored_uuid = existing.nocab_uuid

        binder.register_alias(
            self.SOURCE_GAME, DataSource.POKEMON_TCG, printing_id, stored_uuid
        )

        return stored_uuid if changed else None

    def _find_matching_printing(
        self, name: str, set_code: str, binder: CardBinder
    ) -> GenericCard | None:
        """Find the already-stored card matching (name, set_code), if any.

        Private helper — single consumer is _ingest_row(). Filters
        binder.get_by_name(self.SOURCE_GAME, name) down to the card
        (if any) whose own raw_content["id"] resolves to the same set
        code — see this module's docstring for why name alone isn't
        enough (cross-set same-name cards are distinct cards).

        Inputs:
            name: row["name"] of the row being ingested.
            set_code: this row's own set code (see _set_code()).
            binder: the CardBinder to read from.
        Output: the matching GenericCard, or None if no stored card
            under this name has a matching set code. Assumes at most
            one match — a second candidate with the same (name,
            set_code) is exactly the "duplicate" case _ingest_row()
            is designed to handle, so this method is only ever called
            before that resolution happens, not after.
        Side effects: none.
        Exceptions: none.
        """
        for candidate in binder.get_by_name(self.SOURCE_GAME, name):
            if self._set_code(candidate.raw_content["id"]) == set_code:
                return candidate
        return None

    def _set_code(self, printing_id: str) -> str:
        """Extract the set code portion of a pokemon-tcg-data printing id.

        Private helper — used by both _ingest_row() and
        _find_matching_printing(). E.g. "swsh8-1" -> "swsh8" — see this
        module's docstring for why this (not a "set" field, which is
        always absent in the live data) is the reliable set signal.

        Inputs:
            printing_id: a pokemon-tcg-data "id" field value.
        Output: everything before the last "-" in printing_id.
        Side effects: none.
        Exceptions: none — printing_id is expected to always contain
            at least one "-", confirmed across the full live dataset;
            if it somehow doesn't, rsplit returns printing_id
            unchanged rather than raising.
        """
        return printing_id.rsplit("-", 1)[0]

    def _build_card(self, row: dict, printing_id: str) -> GenericCard:
        """Build a fresh GenericCard for one pokemon-tcg-data row.

        Private helper — single consumer is _ingest_row(), from both
        its create-path and its throwaway-candidate-for-merge path.

        Inputs:
            row: one parsed JSON object from a pokemon-tcg-data
                cards/*.json file's array.
            printing_id: row["id"], passed in rather than re-read.
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
                data_source=DataSource.POKEMON_TCG,
                source_id=printing_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )
