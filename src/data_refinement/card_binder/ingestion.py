"""The shared interface every raw-source ingestion stage implements.

See src/data_refinement/README.md for this container's scope and
src/data_refinement/card_binder/README.md for the full contract this
was built against.

CardIngestionStage is a Strategy (PATTERNS.md) — one implementation
per raw source (e.g. ScryfallCardIngestionStage in ./scryfall/, a
future PokemonTcgCardIngestionStage in ./pokemon_tcg/, etc.). Deliberately
a pure, stateless translator: no filtering, no deduplication, and NO
write access to CardBinder/AliasLedger — a stage only ever returns
data; CardBinder.add()/register_alias() (driven by build.py's driver)
are the only things that ever mutate a CardBinder. This keeps every
stage trivially testable (assert on returned IngestedCandidates, no
CardBinder to construct or mock) and keeps collision/identity logic
centralized in one place for every source, not reimplemented per-stage.

Implemented as a typing.Protocol (structural typing), matching the
convention already established by Metric elsewhere in this project's
architecture, rather than an ABC.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


@dataclass(frozen=True)
class ExternalIdentifier:
    """One (data_source, source_id) pair a raw record carries.

    Used for identifiers OTHER than a candidate's own primary
    Provenance (e.g. a Scryfall row's arena_id alongside its
    oracle_id) — see IngestedCandidate.aliases.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    data_source: DataSource
    source_id: str


@dataclass(frozen=True)
class IngestedCandidate:
    """One raw record translated into a candidate card plus its known aliases.

    aliases holds every identity-bearing field a raw source's row
    carries BEYOND card.provenance's own (data_source, source_id) —
    e.g. a Scryfall row's arena_id/mtgo_id/mtgo_foil_id/each
    multiverse_ids entry, whichever are actually present on that row
    (presence is per-row-optional — see ScryfallCardIngestionStage).

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    card: GenericCard
    aliases: list[ExternalIdentifier]


class CardIngestionStage(Protocol):
    """Strategy: turn one raw source file into candidate cards + their aliases.

    Implementations do not deduplicate or filter, and never touch
    CardBinder/AliasLedger directly — see this module's docstring.
    Single-consumer to build_or_update_card_binder() (build.py), which
    is agnostic to which concrete implementation it receives.
    """

    def ingest(self, raw_path: Path, source_game: GameId) -> list[IngestedCandidate]:
        """Parse raw_path into one IngestedCandidate per raw record.

        Inputs:
            raw_path: path to a raw source file (e.g. a Scryfall
                oracle-cards .jsonl — see
                scryfall/ingestion_stage.py's ScryfallCardIngestionStage).
            source_game: which game these cards belong to.
        Output: one IngestedCandidate per raw record found in raw_path,
            each card with its own freshly minted nocab_uuid. No
            filtering, no deduplication — see this module's docstring.
        Side effects: none — reads raw_path, no other I/O.
        Exceptions: raises if raw_path doesn't exist or isn't
            well-formed for this implementation's expected format.
        """
        ...
