"""Decodes Hearthstone deck codes (deckstrings).

A deckstring is base64 over a list of unsigned LEB128 varints
(HearthSim's documented format, also the in-game "copy deck" code):

    0 (reserved), version (1), format (1 Wild, 2 Standard, 3 Classic,
    4 Twist),
    hero count, hero dbfIds...,
    single-copy count, dbfIds...,
    two-copy count, dbfIds...,
    n-copy count, (dbfId, copies) pairs...,
    then, optionally, 1 and a sideboard section in the same three
    blocks, each entry ending with its owner card's dbfId
    ((dbfId, owner) or (dbfId, copies, owner)).

Sideboard cards (E.T.C., Band Manager's band; Zilliax Deluxe 3000's
modules) are kept apart from the deck's cards: they never sit in the
deck itself.
"""

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator

_SUPPORTED_VERSION = 1
_CONTINUATION_BIT = 0x80
_PAYLOAD_BITS = 0x7F


@dataclass(frozen=True)
class Deckstring:
    """One decoded deck code. Read-only: the mappings are proxies.

    format_id stays the raw int (1 Wild, 2 Standard, 3 Classic,
    4 Twist) rather than an enum, so a format added later still
    decodes. card_counts maps dbfId to copies in the deck (heroes and
    sideboard cards excluded). sideboard_counts maps (owner dbfId,
    dbfId) to copies.
    """

    format_id: int
    hero_dbf_ids: tuple[int, ...]
    card_counts: Mapping[int, int]
    sideboard_counts: Mapping[tuple[int, int], int]

    @property
    def card_count(self) -> int:
        """Total copies in the deck (sideboards excluded).

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return sum(self.card_counts.values())


def decode_deckstring(deck_code: str) -> Deckstring:
    """Decode one deck code.

    Inputs: deck_code (base64 text as the site shows it).
    Output: the Deckstring.
    Side effects: none.
    Exceptions: ValueError if the code isn't base64, is truncated, has
        a nonzero reserved byte or an unsupported version.

    Example:
        >>> decode_deckstring("AAECAaIHCLIC...").card_count
        30
    """
    varints = _read_varints(_decode_base64(deck_code))

    # Header: reserved 0, version, format
    if _next_varint(varints) != 0:
        raise ValueError(f"deck code {deck_code!r} has a nonzero reserved byte")
    version = _next_varint(varints)
    if version != _SUPPORTED_VERSION:
        raise ValueError(f"deck code {deck_code!r} has version {version}")
    format_id = _next_varint(varints)

    # Heroes, then the deck's cards in three copy-count blocks
    hero_dbf_ids = tuple(_next_varint(varints) for _ in range(_next_varint(varints)))
    card_counts = _read_card_blocks(varints)

    # Optional sideboards: a 1 flag, then the same blocks with owners
    sideboard_counts: dict[tuple[int, int], int] = {}
    if _next_varint_or_none(varints) == 1:
        sideboard_counts = _read_sideboard_blocks(varints)

    return Deckstring(
        format_id,
        hero_dbf_ids,
        MappingProxyType(card_counts),
        MappingProxyType(sideboard_counts),
    )


def _decode_base64(deck_code: str) -> bytes:
    """deck_code's bytes, tolerating missing "=" padding.

    Inputs: deck_code. Output: bytes. Side effects: none.
    Exceptions: ValueError if it isn't base64.
    """
    text = deck_code.strip()
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
    except binascii.Error as error:
        raise ValueError(f"deck code {deck_code!r} is not base64: {error}") from error


def _read_varints(data: bytes) -> Iterator[int]:
    """data's unsigned LEB128 varints, in order.

    Inputs: data. Output: an iterator of ints. Side effects: none.
    Exceptions: ValueError (raised lazily) if data ends mid-varint.
    """
    value = 0
    shift = 0
    for byte in data:
        value |= (byte & _PAYLOAD_BITS) << shift
        if byte & _CONTINUATION_BIT:
            shift += 7
            continue
        yield value
        value = 0
        shift = 0
    if shift:
        raise ValueError("deck code ends mid-varint")


def _next_varint(varints: Iterator[int]) -> int:
    """The next varint.

    Inputs: varints. Output: int. Side effects: advances varints.
    Exceptions: ValueError if varints is exhausted (a truncated code).
    """
    result = _next_varint_or_none(varints)
    if result is None:
        raise ValueError("deck code is truncated")
    return result


def _next_varint_or_none(varints: Iterator[int]) -> int | None:
    """The next varint, or None at the end of the code.

    Inputs: varints. Output: int | None. Side effects: advances varints.
    Exceptions: ValueError if data ends mid-varint.
    """
    return next(varints, None)


def _read_card_blocks(varints: Iterator[int]) -> dict[int, int]:
    """The single-copy, two-copy and n-copy blocks as {dbfId: copies}.

    Inputs: varints (positioned at the single-copy count).
    Output: {dbfId: copies}. Side effects: advances varints.
    Exceptions: ValueError on a truncated code.
    """
    result: dict[int, int] = {}
    for copies in (1, 2):
        for _ in range(_next_varint(varints)):
            dbf_id = _next_varint(varints)
            result[dbf_id] = result.get(dbf_id, 0) + copies
    for _ in range(_next_varint(varints)):
        dbf_id = _next_varint(varints)
        result[dbf_id] = result.get(dbf_id, 0) + _next_varint(varints)
    return result


def _read_sideboard_blocks(varints: Iterator[int]) -> dict[tuple[int, int], int]:
    """The sideboard's three blocks as {(owner dbfId, dbfId): copies}.

    Inputs: varints (positioned after the sideboard flag).
    Output: {(owner, dbfId): copies}. Side effects: advances varints.
    Exceptions: ValueError on a truncated code.
    """
    result: dict[tuple[int, int], int] = {}
    for copies in (1, 2):
        for _ in range(_next_varint(varints)):
            dbf_id = _next_varint(varints)
            key = (_next_varint(varints), dbf_id)
            result[key] = result.get(key, 0) + copies
    # An n-copy entry is (dbfId, copies, owner): copies before owner
    for _ in range(_next_varint(varints)):
        dbf_id = _next_varint(varints)
        copies = _next_varint(varints)
        key = (_next_varint(varints), dbf_id)
        result[key] = result.get(key, 0) + copies
    return result
