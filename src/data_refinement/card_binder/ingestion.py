"""The shared interface every raw-source ingestion stage implements.

See src/data_refinement/README.md for this container's scope,
src/data_refinement/card_binder/README.md for the full contract this
was built against, and plans/card_binder_v2.md's "Component overview"
for the reasoning behind this design.

CardIngestionStage is a Strategy (PATTERNS.md) — one implementation
per raw source (e.g. ScryfallCardIngestionStage in ./scryfall/, a
future PokemonTcgCardIngestionStage in ./pokemon_tcg/, etc.). Unlike
the pre-card_binder_v2 design, a stage is NOT a pure, stateless
translator anymore: it's handed a live CardBinder and owns its own
duplicate-detection (however that source's own data supports it — see
each stage's own module docstring) and collision-resolution (typically
one of merge_strategies.py's shared policies) directly against it,
performing create()/update()/replace()/register_alias() calls itself
as a side effect. CardBinder has no opinion on either — see
card_binder.py's own module docstring. This is a deliberate reversal
of the pre-v2 design's centralization: that design kept collision
logic in one place (CardBinder.add()) at the cost of assuming
(source_game, name) was always a valid uniqueness key, which doesn't
hold for every game (Slay the Spire 2's per-character Strike/Defend,
Pokemon's same-named-but-different reprints) — see
plans/card_binder_v2.md's "Why".

A stage always ingests exactly one game — SOURCE_GAME is a class
constant, not an ingest() parameter, so a caller can never construct a
nonsensical call (e.g. passing GameId.GWENT to a Scryfall stage).

Implemented as a typing.Protocol (structural typing), matching the
convention already established by Metric elsewhere in this project's
architecture, rather than an ABC.
"""

from pathlib import Path
from typing import ClassVar, Protocol
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.game_id import GameId


class CardIngestionStage(Protocol):
    """Strategy: parse one raw source and write its cards directly into a CardBinder.

    Single-consumer to build_or_update_card_binder() (build.py), which
    is agnostic to which concrete implementation it receives. See
    src/data_refinement/card_binder/scryfall/ingestion_stage.py for a
    concrete implementation.
    """

    SOURCE_GAME: ClassVar[GameId]

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        """Parse raw_path and create/update/replace cards directly on binder.

        Inputs:
            raw_path: path to a raw source file, or a directory of raw
                source files — interpretation is each implementation's
                own concern.
            binder: the CardBinder to read from and write to, as a
                side effect.
        Output: nocab_uuid of every card this call created or
            content-changed (via update()/replace()) this run. A card
            looked up and left untouched (e.g. an exact-duplicate
            re-fetch) is NOT included.
        Side effects: reads raw_path; creates/updates cards and
            registers aliases directly on binder.
        Exceptions: raises if raw_path doesn't exist or isn't
            well-formed for this implementation's expected format.
        """
        ...
