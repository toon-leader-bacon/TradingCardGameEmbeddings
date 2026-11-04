"""The Unknown sentinel check every DeckExtractionStage needs.

A stage substitutes its game's Unknown sentinel card for a card
reference it can't match (see README.md's "Sources"). The binder must
already hold that card (CardBinder.ensure_unknown_card()), and a stage
only gets a read-only CardLookup, so each stage checks it up front.
Older stages still carry their own copy of this check; new stages use
this one.
"""

from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.game_id import GameId


def require_unknown_card_uuid(card_lookup: CardLookup, source_game: GameId) -> UUID:
    """source_game's Unknown sentinel uuid, checked to be on card_lookup.

    Inputs: card_lookup (read-only binder), source_game.
    Output: the sentinel's nocab_uuid (CardBinder.unknown_card_uuid()).
    Side effects: none.
    Exceptions: RuntimeError naming the missing seeding step.

    Example:
        >>> require_unknown_card_uuid(binder, GameId.POKEMON)
        UUID('...')
    """
    result = CardBinder.unknown_card_uuid(source_game)
    if card_lookup.get_by_uuid(result) is None:
        raise RuntimeError(
            f"{source_game!r}'s Unknown sentinel card is not seeded; call "
            "CardBinder.ensure_unknown_card() before extract()"
        )
    return result
