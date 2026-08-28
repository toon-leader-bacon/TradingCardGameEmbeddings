"""Resolves a 17lands game_data CSV header's card columns to nocab_uuids.

See src/data_refinement/README.md for this container's scope. This is
the one place in game_data_metrics that talks to CardBinder — kept
as its own small, independently testable unit (not inlined into
MetricScanner.scan()) per PRINCIPLES.md's stepwise refinement, since
header resolution is a single, decomposable responsibility distinct
from streaming/accumulating/checkpointing.

17lands' game_data columns are named "<prefix>_<card name>" for five
prefixes per card (opening_hand, drawn, tutored, deck, sideboard).
Most card names resolve directly via CardBinder.get_by_name(). A
minority don't: 17lands names a multi-faced card's columns after only
one face (e.g. "deck_Bruce Banner" for a card whose GenericCard.name is
the combined "Bruce Banner // The Incredible Hulk") — a real, confirmed
mismatch, not hypothetical (see
src/data_refinement/card_binder/README.md for the full story on
CardBinder.get_by_name_regex). The fallback itself (try
get_by_name() first, fall back to get_by_name_regex() with a pattern
matching "the given name, optionally followed by ' // ...'") lives in
the shared name_lookup.py — CardBinder itself stays generic and has no
awareness of this 17lands-specific quirk; the 17lands-specific policy
lives one level up, shared with draft_data_metrics/pick_name_cache.py
rather than duplicated (see tmp/REFACTOR.md §1).
"""

from dataclasses import dataclass
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.name_lookup import find_uuid_by_name
from src.schema.game_id import GameId

_COLUMN_PREFIXES = ("opening_hand", "drawn", "tutored", "deck", "sideboard")
_DECK_COLUMN_PREFIX = "deck_"


@dataclass(frozen=True)
class ColumnResolution:
    """The outcome of resolving one CSV header's card columns.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    resolved: list[CardColumnSet]
    unresolved_names: list[str]  # card names that neither get_by_name()
    # nor the get_by_name_regex() fallback could resolve to exactly
    # one card


def find_card_columns(
    header: list[str], card_binder: CardBinder, source_game: GameId
) -> ColumnResolution:
    """Resolve every card name found in header to a CardColumnSet.

    Composed of: _discover_card_names() to find every distinct card
    name present (via its "deck_<name>" column — the other four
    prefixes are assumed to exist for the same set of names, per
    17lands' file format), then name_lookup.find_uuid_by_name() per
    name, splitting results into ColumnResolution.resolved (built via
    _build_card_column_set()) and .unresolved_names.

    Inputs:
        header: the raw CSV's column names, in file order (e.g. from
            reading just the first row).
        card_binder: registry to resolve card names against.
        source_game: which game's cards to resolve against — 17lands
            game_data is MTG-only today, but this isn't hardcoded here
            since CardBinder itself is generic across GameId.
    Output: a ColumnResolution splitting header's card names into
        successfully resolved CardColumnSets and unresolved names.
    Side effects: none — read-only queries against card_binder.
    Exceptions: none expected from well-formed input.

    Example:
        >>> resolution = find_card_columns(header, registry, GameId.MTG)
        >>> len(resolution.unresolved_names)
    """
    resolved = []
    unresolved_names = []
    for name in _discover_card_names(header):
        nocab_uuid = find_uuid_by_name(card_binder, source_game, name)
        if nocab_uuid is None:
            unresolved_names.append(name)
        else:
            resolved.append(_build_card_column_set(name, nocab_uuid))
    return ColumnResolution(resolved=resolved, unresolved_names=unresolved_names)


def _discover_card_names(header: list[str]) -> list[str]:
    """Extract every distinct card name from header's "deck_<name>" columns.

    Private helper — single consumer is find_card_columns().

    Inputs:
        header: the raw CSV's column names, in file order.
    Output: every card name found via a "deck_<name>" column, in the
        order those columns appear in header.
    Side effects: none.
    Exceptions: none.
    """
    return [
        column[len(_DECK_COLUMN_PREFIX) :]
        for column in header
        if column.startswith(_DECK_COLUMN_PREFIX)
    ]


def _build_card_column_set(name: str, nocab_uuid: UUID) -> CardColumnSet:
    """Build a CardColumnSet's five column names for one resolved card name.

    Private helper — single consumer is find_card_columns(). Pure
    string formatting — no registry access.

    Inputs:
        name: the card name as it appears in the CSV header (i.e. the
            exact string following "deck_", "opening_hand_", etc.).
        nocab_uuid: the nocab_uuid this name resolved to.
    Output: a CardColumnSet with nocab_uuid and all five "<prefix>_
        <name>" column names.
    Side effects: none.
    Exceptions: none.
    """
    columns = {prefix: f"{prefix}_{name}" for prefix in _COLUMN_PREFIXES}
    return CardColumnSet(
        nocab_uuid=nocab_uuid,
        opening_hand=columns["opening_hand"],
        drawn=columns["drawn"],
        tutored=columns["tutored"],
        deck=columns["deck"],
        sideboard=columns["sideboard"],
    )
