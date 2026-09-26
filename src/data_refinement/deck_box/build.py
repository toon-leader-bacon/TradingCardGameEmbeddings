"""Drives one deck extraction run: load, extract, save.

See plans/deck_box.md for the design this implements, and
src/data_refinement/card_binder/build.py for the sibling this was
deliberately modeled on. A plain function, not an orchestrator class —
matches that same "thin runnable snippet" pattern.
"""

from pathlib import Path
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.extraction import DeckExtractionStage


def build_or_update_deck_box(
    raw_path: Path,
    extraction_stage: DeckExtractionStage,
    box_path: Path,
    card_lookup: CardLookup,
) -> list[UUID]:
    """Extract raw_path into box_path, creating or updating it.

    Composed of:
        1. DeckBox.load([box_path]) — connects directly to box_path,
           creating it if it doesn't exist yet (see DeckBox.load()'s
           own docstring). Unlike before this container's SQLite
           rewrite, this is unconditional: single-path load() now
           handles "doesn't exist yet" and "already exists" identically,
           so every deck extraction_stage.extract() creates/updates
           below persists to box_path immediately, as it happens - a
           crash partway through no longer loses progress that hasn't
           reached step 3 yet.
        2. extraction_stage.extract(raw_path, box, card_lookup) —
           creates decks on box, as a side effect, under
           extraction_stage.SOURCE_GAME.
        3. box.save(box_path, extraction_stage.SOURCE_GAME,
           card_lookup.version_for(extraction_stage.SOURCE_GAME)) —
           stamps the box with the CardBinder version its card
           references were just resolved against. Every deck is
           already durable by this point (see step 1) - this call's
           only remaining job is that version stamp.
        4. Return the list[UUID] extract() itself returned, unchanged.

    Inputs:
        raw_path: path to a raw source file (format depends on
            extraction_stage).
        extraction_stage: which DeckExtractionStage to parse raw_path
            with.
        box_path: where the game's deck box file lives — e.g.
            data/final/decks/<game>.db. Connected to directly and
            written to incrementally throughout this call, not just at
            the end (see step 1 above).
        card_lookup: read-only access to an already-populated
            CardBinder for extraction_stage.SOURCE_GAME, passed
            straight through to extraction_stage.extract().
    Output: nocab_uuid of every deck extraction_stage.extract()
        created this run.
    Side effects: creates box_path's parent directory if missing;
        reads raw_path; creates/updates decks directly in box_path as
        extraction proceeds; writes/overwrites box_path's metadata row
        at the end.
    Exceptions: raises whatever extraction_stage.extract(),
        DeckBox.load(), or DeckBox.save() raise.

    Example:
        >>> from src.data_refinement.deck_box.sts_gg.extraction_stage import (
        ...     StsGgDeckExtractionStage,
        ... )
        >>> changed_uuids = build_or_update_deck_box(
        ...     Path("data/raw/sts_gg/runs.jsonl"),
        ...     StsGgDeckExtractionStage(),
        ...     Path("data/final/decks/slay_the_spire_2.db"),
        ...     card_binder,
        ... )
    """
    box = DeckBox.load([box_path])
    changed_uuids = extraction_stage.extract(raw_path, box, card_lookup)
    box.save(
        box_path,
        extraction_stage.SOURCE_GAME,
        card_lookup.version_for(extraction_stage.SOURCE_GAME),
    )
    return changed_uuids
