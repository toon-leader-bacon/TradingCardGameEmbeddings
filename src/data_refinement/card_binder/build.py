"""Drives one ingestion run: load, ingest, save.

See src/data_refinement/README.md for this container's scope and
plans/card_binder_v2.md for the design this implements. Still a plain
function, not an orchestrator class (matches the "thin runnable
snippet" pattern src/data_retrieval/README.md's own examples use).

Much thinner than the pre-card_binder_v2 driver: there is no
per-candidate loop here anymore, and no AddOutcome tallying — all of
that logic now lives inside each CardIngestionStage implementation's
own ingest(), since a stage owns its own identity/collision handling
directly against the CardBinder it's handed (see ingestion.py's module
docstring). This function's only remaining job is the load/save
bracketing around that call, plus knowing which game's binder file to
read/write — which it gets from ingestion_stage.SOURCE_GAME, not a
parameter of its own.
"""

from pathlib import Path
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.ingestion import CardIngestionStage


def build_or_update_card_binder(
    raw_path: Path,
    ingestion_stage: CardIngestionStage,
    binder_path: Path,
) -> list[UUID]:
    """Ingest raw_path into binder_path, creating or updating it.

    Composed of:
        1. CardBinder.load([binder_path]) if binder_path already
           exists, else CardBinder.load([]).
        2. ingestion_stage.ingest(raw_path, binder) — creates/updates
           cards and registers aliases directly on binder, as a side
           effect, under ingestion_stage.SOURCE_GAME.
        3. binder.save(binder_path, ingestion_stage.SOURCE_GAME).
        4. Return the list[UUID] ingest() itself returned, unchanged.

    Inputs:
        raw_path: path to a raw source file (format depends on
            ingestion_stage, e.g. a Scryfall oracle-cards .jsonl for
            ScryfallCardIngestionStage).
        ingestion_stage: which CardIngestionStage to parse raw_path
            with.
        binder_path: where the game's binder file lives (read if
            present, always (re)written at the end) — e.g.
            data/final/cards/<game>.jsonl.
    Output: nocab_uuid of every card ingestion_stage.ingest() created
        or content-changed this run — see CardIngestionStage.ingest()'s
        own docstring for exactly what's included.
    Side effects: reads binder_path if it exists, and raw_path;
        creates binder_path's parent directory if missing; writes/
        overwrites binder_path and its sibling alias_ledger file.
    Exceptions: raises whatever ingestion_stage.ingest(),
        CardBinder.load(), or CardBinder.save() raise.

    Example:
        >>> from src.data_refinement.card_binder.scryfall.ingestion_stage import (
        ...     ScryfallCardIngestionStage,
        ... )
        >>> changed_uuids = build_or_update_card_binder(
        ...     Path("data/raw/scryfall/oracle-cards-20260820090157.jsonl"),
        ...     ScryfallCardIngestionStage(),
        ...     Path("data/final/cards/mtg.jsonl"),
        ... )
    """
    # Start from whatever's already on disk for this game, or empty.
    binder = CardBinder.load([binder_path] if binder_path.exists() else [])

    # The stage owns all identity/collision handling itself, as a side
    # effect against binder — nothing left for this function to loop
    # over or tally.
    changed_uuids = ingestion_stage.ingest(raw_path, binder)

    # Persist the updated binder back under the stage's own game.
    binder.save(binder_path, ingestion_stage.SOURCE_GAME)

    return changed_uuids
