"""The shared card/deck identity envelope every container builds on.

A thin identity envelope (nocab_uuid / provenance / source_game / name)
plus a free-form raw_content blob, rather than a rigid shared schema
that would force semantically different per-game concepts (MTG mana
cost vs. Pokemon HP) into shared fields.

Provenance describes only where the CURRENTLY STORED raw_content came
from; it is not a general identity mapping, since a single card can
carry several external identifiers across several DataSources at once
(e.g. Scryfall's oracle_id AND Arena's numeric id) — that general
mapping is AliasLedger's job instead
(src/data_refinement/card_binder/alias_ledger.py; see
src/data_refinement/card_binder/README.md for the full design).
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from src.schema.data_source import DataSource
from src.schema.game_id import GameId


@dataclass(frozen=True)
class Provenance:
    """Where a GenericCard's CURRENT raw_content snapshot came from.

    Single current snapshot only — no history of past sources a card's
    content has ever come from. A future need for that history would
    be a new, separate design decision, not an extension of this
    dataclass.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    data_source: DataSource
    source_id: str  # that source's own id for the CURRENT raw_content
    fetched_at: datetime


@dataclass
class GenericCard:
    """One card, in any onboarded trading card game.

    raw_content is intentionally untyped past this point — each game's
    CardIngestionStage owns its own shape for it (type, subtype, cost,
    power/toughness, HP, rules text, etc., all un-normalized). The
    embedding model's primary input is expected to be a text
    serialization of raw_content.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: UUID  # this system's own identity
    source_game: GameId
    name: str
    raw_content: dict
    provenance: Provenance


@dataclass
class GenericDeck:
    """One deck, in any onboarded trading card game.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: UUID
    source_game: GameId
    name: str
    card_nocab_uuids: list[
        UUID
    ]  # multiset: unordered, duplicates meaningful (copy count)
