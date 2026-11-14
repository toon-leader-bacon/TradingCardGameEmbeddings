"""Matches Limitless decklist entries to a non-Pokemon game's binder cards.

MATCH ORDER (first hit wins):

    1. PRINTING (number games only): "<set>-<number>" (e.g. "OP13" +
       "001" -> "OP13-001") as a DataSource.PRINTED_NUMBER alias, which
       the tcgjson and tcgcsv binders register for every printed number.
    2. NAME: fold_card_name(key.name) against the same key of every
       binder card. A key several binder cards share is ambiguous
       (Digimon prints many distinct "Agumon") and matches nothing.
       Pokemon's card_index.py instead takes the newest same-name card;
       see its docstring for why that holds there.
    3. UNMATCHED: None; the caller substitutes the Unknown sentinel.

Caching and match counts: CachedCardMatcher (deck_box/card_matching.py).

fold_card_name is not tcgjson's printing_groups._identity_key, though
both read ", " as " - ". That key groups TCGplayer product names (it
strips printing tags such as "(Foil)"). This one compares two finished
spellings of a card's name, the binder's and Limitless's, which also
differ in diacritics ("Te Kā"), zero-width spaces and quotes.
"""

import re
import unicodedata
from typing import Iterable
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.card_matching import (
    CachedCardMatcher,
    CardMatchKind,
)
from src.data_refinement.deck_box.limitless_play.limitless_decklist import (
    LimitlessCardRef,
)
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# Zero-width characters dropped; curly quotes straightened
_CHARACTER_FOLDS = str.maketrans(
    {
        "\u200b": None,  # zero-width space
        "\u200c": None,  # zero-width non-joiner
        "\u200d": None,  # zero-width joiner
        "\ufeff": None,  # zero-width no-break space
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)
# ", " and " - " between a name and its subtitle
_SUBTITLE_SEPARATOR = re.compile(r"\s*,\s+|\s+-\s+")
_WHITESPACE_RUN = re.compile(r"\s+")


class LimitlessTcgCardIndex(CachedCardMatcher[LimitlessCardRef]):
    """Limitless (set code, number, name) -> binder nocab_uuid, cached.

    Single-consumer to
    tcg_extraction_stages.py.
    """

    def __init__(
        self,
        game: GameId,
        card_lookup: CardLookup,
        uuid_by_folded_name: dict[str, UUID | None],
        match_by_number: bool,
    ) -> None:
        """
        Inputs: game, card_lookup (the game's binder, read-only),
            uuid_by_folded_name (fold_card_name -> nocab_uuid, None for an
            ambiguous key; see index_names), match_by_number.
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        super().__init__()
        self._game = game
        self._card_lookup = card_lookup
        self._uuid_by_folded_name = uuid_by_folded_name
        self._match_by_number = match_by_number

    @classmethod
    def from_card_lookup(
        cls, game: GameId, card_lookup: CardLookup, match_by_number: bool
    ) -> "LimitlessTcgCardIndex":
        """Build an index over every binder card of game.

        Inputs: game, card_lookup, match_by_number.
        Output: LimitlessTcgCardIndex.
        Side effects: reads every card of game once. The Unknown
            sentinel is left out, so no entry matches it by name.
        Exceptions: none.

        Example:
            >>> LimitlessTcgCardIndex.from_card_lookup(
            ...     GameId.ONE_PIECE, binder, match_by_number=True
            ... )
        """
        unknown_uuid = CardBinder.unknown_card_uuid(game)
        real_cards = (
            card
            for card in card_lookup.all_cards(game)
            if card.nocab_uuid != unknown_uuid
        )
        return cls(game, card_lookup, index_names(real_cards), match_by_number)

    def _match(self, key: LimitlessCardRef) -> tuple[UUID | None, CardMatchKind]:
        """Steps 1 to 3 of the module docstring's MATCH ORDER.

        Inputs: key. Output: (nocab_uuid or None, the step that hit).
        Side effects: none (card_lookup reads only). Exceptions: none.
        """
        if self._match_by_number:
            card_uuid = self._uuid_by_printed_number(key)
            if card_uuid is not None:
                return card_uuid, CardMatchKind.PRINTING
        card_uuid = self._uuid_by_folded_name.get(fold_card_name(key.name))
        if card_uuid is not None:
            return card_uuid, CardMatchKind.NAME
        return None, CardMatchKind.UNMATCHED

    def _uuid_by_printed_number(self, key: LimitlessCardRef) -> UUID | None:
        """Step 1: "<set code>-<number>" as a PRINTED_NUMBER alias.

        Inputs: key. Output: nocab_uuid or None.
        Side effects: none. Exceptions: none.
        """
        card = self._card_lookup.get_by_alias(
            self._game, DataSource.PRINTED_NUMBER, f"{key.set_code}-{key.number}"
        )
        return card.nocab_uuid if card is not None else None


def index_names(cards: Iterable[GenericCard]) -> dict[str, UUID | None]:
    """fold_card_name -> the one card with that key, or None when several
    distinct cards share it.

    Inputs: cards (one game's binder cards). Output: dict.
    Side effects: none. Exceptions: none.

    Example:
        >>> index_names([agumon_bt1, agumon_bt5, gabumon])
        {'agumon': None, 'gabumon': UUID('...')}
    """
    result: dict[str, UUID | None] = {}
    for card in cards:
        folded_name = fold_card_name(card.name)
        if folded_name in result and result[folded_name] != card.nocab_uuid:
            result[folded_name] = None
        else:
            result.setdefault(folded_name, card.nocab_uuid)
    return result


def fold_card_name(name: str) -> str:
    """name folded for comparing two spellings of one card name.

    NFKD with combining marks dropped, zero-width characters dropped,
    curly quotes straightened, ", " and " - " read as one separator,
    whitespace runs collapsed, casefolded.

    Inputs: name. Output: str. Side effects: none. Exceptions: none.

    Example:
        >>> fold_card_name("Sabine Wren, Explosives Artist")
        'sabine wren - explosives artist'
        >>> fold_card_name("Te Kā - Heartless") == fold_card_name("Te Ka - Heartless")
        True
    """
    decomposed = unicodedata.normalize("NFKD", name)
    result = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).translate(_CHARACTER_FOLDS)
    result = _SUBTITLE_SEPARATOR.sub(" - ", result)
    return _WHITESPACE_RUN.sub(" ", result).strip().casefold()
