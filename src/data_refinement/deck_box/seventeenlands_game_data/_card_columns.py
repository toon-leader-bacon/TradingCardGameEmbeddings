"""Resolving a game_data CSV's header-derived deck_<name> columns to nocab_uuids.

Single consumer: src/data_refinement/deck_box/seventeenlands_game_data/extraction_stage.py.
See that module's docstring (CARD MATCHING and UNRESOLVED CARDS
sections) for the matching/fallback policy these helpers implement.
"""

import logging
from typing import Iterable
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import (
    CardLookup,
    uuid_for_name_or_front_face,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_DECK_COLUMN_PREFIX = "deck_"


def _deck_column_index(
    header_columns: Iterable[str], card_lookup: CardLookup, source_game: GameId
) -> list[tuple[str, UUID]]:
    """Match every "deck_<name>" column in header_columns to a nocab_uuid.

    Private helper — single consumer is extraction_stage.py's
    _extract_csv(). Every distinct <name> is resolved at most once per
    call (a local cache), via _card_uuid_for_name().

    Inputs:
        header_columns: this CSV's own header (e.g.
            pandas.read_csv(path, nrows=0).columns) — every column,
            not just deck_-prefixed ones; non-matching columns are
            ignored.
        card_lookup: registry each distinct <name> is resolved
            against.
        source_game: the game whose Unknown sentinel card backs an
            unresolved name (see _card_uuid_for_name()).
    Output: every "deck_<name>" column paired with its resolved (or
        Unknown-sentinel-substituted — see extraction_stage.py's module
        docstring's UNRESOLVED CARDS section) nocab_uuid, in
        header_columns' own order.
    Side effects: emits one logging.error() per distinct
        unresolved <name> (via _card_uuid_for_name()).
    Exceptions: raises RuntimeError if source_game's Unknown sentinel
        card isn't found on card_lookup.
    """
    name_cache: dict[str, UUID] = {}
    deck_columns: list[tuple[str, UUID]] = []
    for column in header_columns:
        if not column.startswith(_DECK_COLUMN_PREFIX):
            continue
        name = column[len(_DECK_COLUMN_PREFIX) :]
        if name not in name_cache:
            name_cache[name] = _card_uuid_for_name(name, card_lookup, source_game)
        deck_columns.append((column, name_cache[name]))
    return deck_columns


def _card_uuid_for_name(
    name: str, card_lookup: CardLookup, source_game: GameId
) -> UUID:
    """Resolve one bare card name to a nocab_uuid, falling back to Unknown.

    Private helper — single consumer is _deck_column_index().
    uuid_for_name_or_front_face(), then the Unknown sentinel — see
    extraction_stage.py's module docstring's CARD MATCHING and
    UNRESOLVED CARDS sections.

    Inputs:
        name: one deck_<name> column's bare <name> suffix.
        card_lookup: registry to resolve name against.
        source_game: the game whose Unknown sentinel card backs a miss.
    Output: the matching nocab_uuid, or (on a miss) the Unknown
        sentinel's nocab_uuid.
    Side effects: emits one logging.error() call on a miss.
    Exceptions: raises RuntimeError if even the Unknown sentinel
        isn't found on card_lookup.
    """
    card_uuid = uuid_for_name_or_front_face(card_lookup, source_game, name)
    if card_uuid is not None:
        return card_uuid

    _logger.error(
        "SeventeenLandsGameDataDeckExtractionStage: unresolved card name "
        "%r — substituting the Unknown sentinel card",
        name,
    )
    unknown_card = card_lookup.get_by_name_single(
        source_game, CardBinder.UNKNOWN_CARD_NAME, strict=False
    )
    if unknown_card is None:
        raise RuntimeError(
            f"SeventeenLandsGameDataDeckExtractionStage: {source_game!r}'s "
            "Unknown sentinel card is not seeded — call "
            "CardBinder.ensure_unknown_card() before extract()"
        )
    return unknown_card.nocab_uuid
