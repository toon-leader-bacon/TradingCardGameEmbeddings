"""Matches crawled MTG deck entries to MTG binder cards.

MATCH ORDER (first hit wins):

    1. CARD_ID: the oracle id as a DataSource.SCRYFALL alias, the
       binder's primary alias (Archidekt).
    2. PRINTING: the printing id as a DataSource.SCRYFALL_PRINTING alias
       (Moxfield, Deckstats). The Scryfall stage registers these from
       the default-cards dump; a binder built before that has none, and
       entries fall through to step 3.
    3. NAME: uuid_for_name_or_front_face (card_lookup.py): the exact
       name, either face of a two-faced card, or a PRINTED_NAME alias;
       never a guess when several cards fit.
    4. UNMATCHED: None; the caller substitutes the Unknown sentinel.
"""

from uuid import UUID

from src.data_refinement.card_binder.card_lookup import (
    CardLookup,
    uuid_for_name_or_front_face,
)
from src.data_refinement.deck_box.card_matching import (
    CachedCardMatcher,
    CardMatchKind,
)
from src.data_refinement.deck_box.mtg_crawls.crawled_deck import MtgCardRef
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class MtgCardMatcher(CachedCardMatcher[MtgCardRef]):
    """MtgCardRef -> MTG binder nocab_uuid, cached and counted.

    Single-consumer to CrawledMtgDeckExtractionStage.
    """

    def __init__(self, card_lookup: CardLookup) -> None:
        """
        Inputs: card_lookup (the MTG binder, read-only).
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        super().__init__()
        self._card_lookup = card_lookup

    def _match(self, key: MtgCardRef) -> tuple[UUID | None, CardMatchKind]:
        """Steps 1 to 4 of the module docstring's MATCH ORDER.

        Inputs: key. Output: (nocab_uuid or None, the step that hit).
        Side effects: none (card_lookup reads only). Exceptions: none.
        """
        if key.oracle_id is not None:
            card = self._card_lookup.get_by_alias(
                GameId.MTG, DataSource.SCRYFALL, key.oracle_id
            )
            if card is not None:
                return card.nocab_uuid, CardMatchKind.CARD_ID
        if key.printing_id is not None:
            card = self._card_lookup.get_by_alias(
                GameId.MTG, DataSource.SCRYFALL_PRINTING, key.printing_id
            )
            if card is not None:
                return card.nocab_uuid, CardMatchKind.PRINTING
        card_uuid = uuid_for_name_or_front_face(self._card_lookup, GameId.MTG, key.name)
        if card_uuid is not None:
            return card_uuid, CardMatchKind.NAME
        return None, CardMatchKind.UNMATCHED
