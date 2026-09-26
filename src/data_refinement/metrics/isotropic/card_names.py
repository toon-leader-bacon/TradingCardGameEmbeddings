"""Card-name -> nocab_uuid lookup shared by both isotropic subpackages
(summary/ Flavor A, games/ Flavor B).

Both flavors spell cards by name and look them up the same way, against
GameId.DOMINION, so this lives at their shared parent rather than in
either one. Each subpackage's own logic (Flavor A's kingdom-constraint
filtering, Flavor B's header parsing) stays in that subpackage.
"""

from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.game_id import GameId


def card_uuid_for_name(card_lookup: CardLookup, card_name: str) -> UUID | None:
    """Look up an isotropic card name against the dominiontabs CardBinder.

    Inputs:
        card_lookup: already-populated registry - must already have
            dominiontabs' cards ingested (this function never writes to
            it). Resolution is by NAME (GameId.DOMINION), not by alias
            - isotropic's raw data (either flavor) was never itself
            ingested into any CardBinder, so there is no
            DataSource.ISOTROPIC alias to join through; the card names
            isotropic's own data already spells out (e.g. "Council
            Room") are assumed to match dominiontabs' own card names
            directly.
        card_name: one card name as it appears in isotropic data (a
            Flavor A board.supply/end.deck entry, or a Flavor B
            GameHeader field).
    Output: the matching nocab_uuid, or None if no dominiontabs card is
        registered under this exact name.
    Side effects: none.
    Exceptions: raises whatever CardBinder.get_by_name_single(strict=True)
        raises if card_name is ambiguous under GameId.DOMINION (more
        than one dominiontabs card sharing that exact name) - that
        exception is deliberately allowed to propagate, not caught
        here, since an ambiguous name is a real data-integrity problem
        worth surfacing loudly rather than papering over.

    Example:
        >>> card_uuid_for_name(card_binder, "Council Room")
        UUID('...')
    """
    card = card_lookup.get_by_name_single(GameId.DOMINION, card_name)
    return card.nocab_uuid if card is not None else None
