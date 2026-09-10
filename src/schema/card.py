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


@dataclass(frozen=True)
class GenericCard:
    """One card, in any onboarded trading card game.

    raw_content is intentionally untyped past this point — each game's
    CardIngestionStage owns its own shape for it (type, subtype, cost,
    power/toughness, HP, rules text, etc., all un-normalized). The
    embedding model's primary input is expected to be a text
    serialization of raw_content.

    frozen=True only blocks reassigning a field on an existing instance
    (`card.name = ...`) — it does NOT make raw_content itself immutable,
    since raw_content is still a plain, mutable dict. Mutating it in
    place (`card.raw_content[key] = ...`) silently corrupts every other
    holder of the same GenericCard object (e.g. CardBinder's own
    canonical copy, if a caller ever gets a reference to it) without
    frozen catching it. Never mutate raw_content in place — build a new
    dict and use dataclasses.replace() to get a new GenericCard instead.
    CardBinder's own read accessors (get_by_uuid() etc.) return a deep
    copy for exactly this reason, so a caller mutating what it got back
    can't reach the binder's own stored card either way.

    One more frozen gotcha: frozen=True + the default eq=True makes
    Python auto-generate a __hash__ for this class, but that __hash__
    will raise TypeError the moment it's actually called, since
    raw_content (a dict) isn't hashable. Don't put a GenericCard in a
    set or use one as a dict key — it isn't genuinely hashable despite
    looking like it should be.

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


@dataclass(frozen=True)
class GenericDeck:
    """One deck, in any onboarded trading card game.

    frozen=True only blocks reassigning a field on an existing instance
    — card_nocab_uuids is still a plain, mutable list, so the same
    in-place-mutation caveat GenericCard's docstring describes for
    raw_content applies here too (e.g. don't `deck.card_nocab_uuids
    .append(...)` on a shared instance; use dataclasses.replace()).
    DeckBox's own read accessors (get_by_uuid() etc.) return a deep
    copy for the same reason CardBinder's do.

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
