"""The shared "try CardBinder.get_by_name() first, fall back to
get_by_name_regex() for a multi-faced card, ambiguous match = unresolved"
policy — the one 17lands-specific by-NAME resolution rule, needed by more
than one pipeline (game_data_metrics/column_lookup.py's find_card_columns()
and draft_data_metrics/pick_name_cache.py's PickNameCache both implement
this identically today; see tmp/REFACTOR.md §1). Lives at this
seventeenlands/ shared level, not duplicated in each consumer.

Distinct from lookup_cache.py's LookupCache: this module is the POLICY
(which CardBinder calls to make, in what order, what counts as ambiguous),
with no caching of its own. column_lookup.py calls it directly per name
(header resolution runs once, nothing worth caching); pick_name_cache.py's
PickNameCache wraps it in a LookupCache (a handful of distinct pick names
recur across millions of rows, so caching pays for itself there).
"""

import re
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.game_id import GameId


def find_uuid_by_name(
    card_binder: CardBinder, source_game: GameId, name: str
) -> UUID | None:
    """Resolve one card name to a nocab_uuid, with a multi-faced-card fallback.

    Tries card_binder.get_by_name(source_game, name) first. On a miss,
    falls back to card_binder.get_by_name_regex(source_game,
    f"^{re.escape(name)}( //.*)?$") — if that returns exactly one card,
    resolves to it; if it returns zero or more than one (ambiguous),
    this name is unresolved. Ambiguity is treated the same as no match
    at all — never guesses among multiple candidates.

    Inputs:
        card_binder: registry to resolve against.
        source_game: which game's cards to resolve against.
        name: a card name to resolve (from a CSV header column or a
            per-row cell value — this function has no opinion on
            where name came from).
    Output: the resolved nocab_uuid, or None if unresolved (by either
        method).
    Side effects: none — read-only queries against card_binder.
    Exceptions: none expected.

    Example:
        >>> find_uuid_by_name(card_binder, GameId.MTG, "Bruce Banner")
    """
    exact = card_binder.get_by_name(source_game, name)
    if exact is not None:
        return exact.nocab_uuid

    fallback_matches = card_binder.get_by_name_regex(
        source_game, f"^{re.escape(name)}( //.*)?$"
    )
    if len(fallback_matches) == 1:
        return fallback_matches[0].nocab_uuid
    return None
