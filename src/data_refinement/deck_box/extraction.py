"""The shared interface every raw-source deck extraction stage implements.

See src/data_refinement/deck_box/README.md for the design this
implements, and src/data_refinement/card_binder/ingestion.py for the
sibling this was deliberately modeled on.

DeckExtractionStage is a Strategy (PATTERNS.md) — one implementation
per raw deck source. Unlike CardIngestionStage, a stage here never
mutates cards: it's handed a CardLookup (read-only) to resolve raw
card references into nocab_uuids, and a live DeckBox to create()/
update() GenericDecks into. There is no collision/identity-merge
concern for a stage to own here the way CardBinder has — a raw row is
usually a distinct new deck, so create() is the common case. A stage
MAY call update() instead, when it derives a stable, deterministic
identity from the raw source's own id (e.g. via uuid5) to make re-runs
idempotent rather than duplicating a deck on every re-run — see
src/data_refinement/deck_box/sts_gg/extraction_stage.py for the
concrete example. replace() is not used by any stage today.

A stage's own extract() is also free to widen raw_path's declared type
to Path | None, defaulting to its own DEFAULT_RAW_PATH-style class
constant when omitted — see the sts_gg stage's use of this convention.
This is a Protocol-safe widening (accepting more than the Protocol
requires), not a contract change.

A stage always extracts decks for exactly one game — SOURCE_GAME is a
class constant, not an extract() parameter, matching
CardIngestionStage's own reasoning.

Implemented as a typing.Protocol (structural typing), matching the
convention CardIngestionStage and seventeenlands' Metric already
established.
"""

from pathlib import Path
from typing import ClassVar, Protocol
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.game_id import GameId


class DeckExtractionStage(Protocol):
    """Strategy: parse one raw source and write its decks directly into a DeckBox.

    Single-consumer to build_or_update_deck_box() (build.py), which is
    agnostic to which concrete implementation it receives.
    """

    SOURCE_GAME: ClassVar[GameId]

    def extract(
        self, raw_path: Path, box: DeckBox, card_lookup: CardLookup
    ) -> list[UUID]:
        """Parse raw_path and create decks directly on box.

        Inputs:
            raw_path: path to a raw source file, or a directory of raw
                source files — interpretation is each implementation's
                own concern.
            box: the DeckBox to write to, as a side effect.
            card_lookup: read-only access to an already-populated
                CardBinder for SOURCE_GAME, used to resolve raw card
                references into nocab_uuids. This stage never
                creates/updates/replaces cards, so CardLookup (not a
                full CardBinder) is the correct, least-privilege type
                — see card_binder/README.md's own reasoning for
                Dojo.prepare_splits.
        Output: nocab_uuid of every deck this call created this run.
        Side effects: reads raw_path; creates decks directly on box.
        Exceptions: raises if raw_path doesn't exist or isn't
            well-formed for this implementation's expected format.
        """
        ...
