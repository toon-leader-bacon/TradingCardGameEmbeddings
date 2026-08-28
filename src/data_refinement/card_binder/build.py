"""Drives one ingestion run: load, ingest, merge, save, summarize.

See src/data_refinement/README.md for this container's scope.
Renamed from build_or_update_card_registry — still a plain function,
not a new orchestrator class (matches the "thin runnable snippet"
pattern src/data_retrieval/README.md's examples already use).
"""

from dataclasses import dataclass
from pathlib import Path

from src.data_refinement.card_binder.card_binder import AddOutcome, CardBinder
from src.data_refinement.card_binder.ingestion import CardIngestionStage
from src.schema.game_id import GameId


@dataclass(frozen=True)
class IngestSummary:
    """Tallied AddOutcome counts from one build_or_update_card_binder() run.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    inserted: int
    content_updated: int
    kept_existing: int


def build_or_update_card_binder(
    raw_path: Path,
    source_game: GameId,
    ingestion_stage: CardIngestionStage,
    binder_path: Path,
) -> IngestSummary:
    """Ingest raw_path into binder_path, creating or updating it.

    Composed of:
        1. CardBinder.load([binder_path]) if binder_path already
           exists, else CardBinder.load([]).
        2. ingestion_stage.ingest(raw_path, source_game) — produces
           IngestedCandidates from the raw source.
        3. Per IngestedCandidate: one binder.add(candidate.card) call,
           tallied into IngestSummary's counters, followed by one
           binder.register_alias(...) call per entry in
           candidate.aliases (using the nocab_uuid add() actually
           stored the card under — see AddResult.stored_card — not
           necessarily candidate.card's own nocab_uuid, if it lost a
           collision).
        4. binder.save(binder_path, source_game).
        5. Return the IngestSummary built in step 3.

    Inputs:
        raw_path: path to a raw source file (format depends on
            ingestion_stage, e.g. a Scryfall oracle-cards .jsonl for
            ScryfallCardIngestionStage).
        source_game: which game raw_path's cards belong to.
        ingestion_stage: which CardIngestionStage to parse raw_path
            with (dependency injection — PATTERNS.md).
        binder_path: where the game's binder file lives (read if
            present, always (re)written at the end) — e.g.
            data/final/cards/<game>.jsonl.
    Output: an IngestSummary tallying how many candidates were
        INSERTED, CONTENT_UPDATED, or KEPT_EXISTING.
    Side effects: reads binder_path if it exists, and raw_path;
        creates binder_path's parent directory if missing; writes/
        overwrites binder_path and its sibling alias_ledger file.
    Exceptions: raises whatever ingestion_stage.ingest(),
        CardBinder.load(), or CardBinder.save() raise.

    Example:
        >>> from src.data_refinement.card_binder.scryfall.ingestion_stage import (
        ...     ScryfallCardIngestionStage,
        ... )
        >>> summary = build_or_update_card_binder(
        ...     Path("data/raw/scryfall/oracle-cards-20260820090157.jsonl"),
        ...     GameId.MTG,
        ...     ScryfallCardIngestionStage(),
        ...     Path("data/final/cards/mtg.jsonl"),
        ... )
        >>> summary.inserted
    """
    binder = CardBinder.load([binder_path] if binder_path.exists() else [])
    candidates = ingestion_stage.ingest(raw_path, source_game)

    inserted = 0
    content_updated = 0
    kept_existing = 0
    for candidate in candidates:
        result = binder.add(candidate.card)
        if result.outcome == AddOutcome.INSERTED:
            inserted += 1
        elif result.outcome == AddOutcome.CONTENT_UPDATED:
            content_updated += 1
        else:
            kept_existing += 1

        for alias in candidate.aliases:
            binder.register_alias(
                candidate.card.source_game,
                alias.data_source,
                alias.source_id,
                result.stored_card.nocab_uuid,
            )

    binder.save(binder_path, source_game)

    return IngestSummary(
        inserted=inserted,
        content_updated=content_updated,
        kept_existing=kept_existing,
    )
