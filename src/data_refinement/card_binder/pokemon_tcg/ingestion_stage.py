"""Translates pokemon-tcg-data per-set card dumps into candidate cards.

See src/data_refinement/card_binder/ingestion.py for the shared
CardIngestionStage interface this implements, and this container's
README for the raw source this reads
(data/raw/pokemon_tcg/cards/*.json, written by
src/data_retrieval/pokemon_tcg/downloader.py).

Unlike ScryfallCardIngestionStage (../scryfall/ingestion_stage.py),
which reads one .jsonl file, this stage reads a *directory* of .json
files — one per set, each a JSON array of card objects — since
pokemon-tcg-data ships that way and has no single oracle_id-equivalent
field pre-deduplicating printings to one row per card identity. Every
element across every file is treated as its own printing;
CardBinder.add()'s existing (source_game, name) richness comparison is
what collapses same-named reprints into one canonical card — no new
collision logic is introduced here.

Confirmed by sampling data/raw/pokemon_tcg/cards/base4.json directly:
  - "id" (e.g. "base4-1") is printing-specific (set code + card
    number) — used as Provenance.source_id, the closest analog to
    Scryfall's oracle_id this data has, even though (unlike oracle_id)
    it does not identify "the same card" across reprints.
  - "nationalPokedexNumbers" identifies a Pokémon *species*, not a
    printing — deliberately NOT extracted as an alias (see
    plans/pokemon_tcg_ingestion.md): many distinct printings share the
    same dex number, and AliasLedger.register() has no defined
    multi-owner behavior for one alias key.
  - No other field plays a role comparable to Scryfall's
    arena_id/mtgo_id/multiverse_ids, so IngestedCandidate.aliases is
    always [] for this source.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.ingestion import IngestedCandidate
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class PokemonTcgCardIngestionStage:
    """Translates a pokemon-tcg-data cards/ directory into IngestedCandidates.

    Single-consumer to src/data_refinement/card_binder/ — no other
    container depends on this class directly (they go through
    CardIngestionStage/build_or_update_card_binder instead).
    """

    def ingest(self, raw_path: Path, source_game: GameId) -> list[IngestedCandidate]:
        """Parse every *.json file under raw_path into IngestedCandidates.

        One IngestedCandidate per card object across every *.json
        file directly under raw_path (each file is a JSON array), via
        _parse_card(). No filtering, no deduplication — see
        src/data_refinement/card_binder/ingestion.py's module
        docstring.

        Inputs:
            raw_path: path to a directory of pokemon-tcg-data per-set
                .json files (e.g. data/raw/pokemon_tcg/cards, written
                by src/data_retrieval/pokemon_tcg/downloader.py). Not
                a single file — see this module's docstring.
            source_game: which game these cards belong to (expected to
                be GameId.POKEMON for this implementation, but not
                enforced).
        Output: one IngestedCandidate per card object found across
            every *.json file directly under raw_path.
        Side effects: none — reads raw_path's *.json files, no other
            I/O.
        Exceptions: raises if raw_path doesn't exist, isn't a
            directory, contains no *.json files, or a file isn't a
            JSON array of objects each carrying "id" and "name".

        Example:
            >>> stage = PokemonTcgCardIngestionStage()
            >>> candidates = stage.ingest(
            ...     Path("data/raw/pokemon_tcg/cards"),
            ...     GameId.POKEMON,
            ... )
        """
        if not raw_path.is_dir():
            raise ValueError(f"pokemon_tcg ingest: {raw_path} is not a directory")

        set_paths = sorted(raw_path.glob("*.json"))
        if not set_paths:
            raise ValueError(f"pokemon_tcg ingest: no *.json files found under {raw_path}")

        candidates = []
        for set_path in set_paths:
            with open(set_path, "r", encoding="utf-8") as set_file:
                rows = json.load(set_file)
            for row in rows:
                candidates.append(self._parse_card(row, source_game))
        return candidates

    def _parse_card(self, row: dict, source_game: GameId) -> IngestedCandidate:
        """Translate one pokemon-tcg-data card object into an IngestedCandidate.

        Private helper — single consumer is ingest(). Composed of:
        building a GenericCard (nocab_uuid=uuid4(), source_game, name=
        row["name"], raw_content=row, provenance=Provenance(
        data_source=DataSource.POKEMON_TCG, source_id=row["id"],
        fetched_at=<now>)) with IngestedCandidate.aliases always [] —
        see this module's docstring for why no aliases are extracted
        for this source.

        Inputs:
            row: one parsed JSON object from a pokemon-tcg-data
                cards/*.json file's array.
            source_game: which game this row belongs to.
        Output: an IngestedCandidate wrapping the translated
            GenericCard, with an empty aliases list.
        Side effects: none.
        Exceptions: raises if row is missing "id" or "name".
        """
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=source_game,
            name=row["name"],
            raw_content=row,
            provenance=Provenance(
                data_source=DataSource.POKEMON_TCG,
                source_id=row["id"],
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        return IngestedCandidate(card=card, aliases=[])
