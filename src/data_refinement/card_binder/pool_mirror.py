"""Shared steps of a stage whose source is a game's whole card pool, so
the binder mirrors it: store each incoming card on the stored card it
replaces (or as a new one), then delete the game's cards nothing took.

Each stage keeps what is source-specific: which stored card a row
replaces (its alias lookup), how a candidate card is built, and which
aliases it registers. Used by ygoprodeck/ingestion_stage.py;
hearthstonejson/ and tcgjson/ still carry their own copies of these two
steps (card_binder/TODO.md).
"""

from uuid import UUID

from src.data_refinement.card_binder import merge_strategies
from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard
from src.schema.game_id import GameId


def store_mirrored_card(
    binder: CardBinder, existing: GenericCard | None, candidate: GenericCard
) -> tuple[UUID, bool]:
    """Create candidate, or merge it onto existing with
    merge_strategies.keep_incoming_if_content_differs (the source is
    authoritative; existing keeps its nocab_uuid).

    Inputs: binder (CardBinder), existing (GenericCard | None, the
        stored card candidate replaces; None to create), candidate
        (GenericCard, freshly built).
    Output: (stored nocab_uuid, whether the card was created or its
        content changed).
    Side effects: creates or replaces one card in binder.
    Exceptions: none beyond CardBinder's own.

    Example:
        >>> store_mirrored_card(binder, None, card)
        (card.nocab_uuid, True)
    """
    if existing is None:
        binder.create(candidate)
        return candidate.nocab_uuid, True
    merged = merge_strategies.keep_incoming_if_content_differs(existing, candidate)
    changed = merged != existing
    if changed:
        binder.replace(existing.nocab_uuid, merged)
    return existing.nocab_uuid, changed


def delete_cards_except(binder: CardBinder, game: GameId, keep: set[UUID]) -> None:
    """Delete every card of game not in keep; the Unknown sentinel is
    always kept.

    Inputs: binder (CardBinder), game (GameId), keep (set[UUID], the
        cards this ingest stored).
    Output: none.
    Side effects: deletes cards from binder (their aliases stay in the
        ledger and read as no match).
    Exceptions: none beyond CardBinder's own.

    Example:
        >>> delete_cards_except(binder, GameId.YUGIOH, taken)
    """
    kept = keep | {CardBinder.unknown_card_uuid(game)}
    stale = [uuid for uuid in binder.all_uuids(game) if uuid not in kept]
    for uuid in stale:
        binder.delete(uuid)
