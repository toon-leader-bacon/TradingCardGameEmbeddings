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
        1. DeckBox.load([box_path]) if box_path already exists, else
           DeckBox.load([]).
        2. extraction_stage.extract(raw_path, box, card_lookup) —
           creates decks on box, as a side effect, under
           extraction_stage.SOURCE_GAME.
        3. box.save(box_path, extraction_stage.SOURCE_GAME).
        4. Return the list[UUID] extract() itself returned, unchanged.

    Inputs:
        raw_path: path to a raw source file (format depends on
            extraction_stage).
        extraction_stage: which DeckExtractionStage to parse raw_path
            with.
        box_path: where the game's deck box file lives (read if
            present, always (re)written at the end) — e.g.
            data/final/decks/<game>.jsonl.
        card_lookup: read-only access to an already-populated
            CardBinder for extraction_stage.SOURCE_GAME, passed
            straight through to extraction_stage.extract().
    Output: nocab_uuid of every deck extraction_stage.extract()
        created this run.
    Side effects: reads box_path if it exists, and raw_path; creates
        box_path's parent directory if missing; writes/overwrites
        box_path.
    Exceptions: raises whatever extraction_stage.extract(),
        DeckBox.load(), or DeckBox.save() raise.

    Example:
        >>> from src.data_refinement.deck_box.sts_gg.extraction_stage import (
        ...     StsGgDeckExtractionStage,
        ... )
        >>> changed_uuids = build_or_update_deck_box(
        ...     Path("data/raw/sts_gg/runs.jsonl"),
        ...     StsGgDeckExtractionStage(),
        ...     Path("data/final/decks/slay_the_spire_2.jsonl"),
        ...     card_binder,
        ... )
    """
    box = DeckBox.load([box_path] if box_path.exists() else [])
    changed_uuids = extraction_stage.extract(raw_path, box, card_lookup)
    box.save(box_path, extraction_stage.SOURCE_GAME)
    return changed_uuids
