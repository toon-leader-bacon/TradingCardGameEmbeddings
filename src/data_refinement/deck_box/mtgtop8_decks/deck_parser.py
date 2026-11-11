"""Reads one MTGTop8 crawl row (src/data_retrieval/mtgtop8/) into a
CrawledDeck.

RAW SHAPE (the fields read):
    {"deck_id": int, "format": "<site format code>", "deck_name": str,
     "mtgo": "<MTGO text export>", ...}
The export is "<count> <name>" lines (CRLF), the sideboard after a
"Sideboard" line.

WHAT COUNTS AS THE DECK: the main section, plus, by the format's
SideboardRole (_SIDEBOARD_ROLES):
  - COMMAND_ZONE (the Commander formats): the whole "Sideboard"
    section: MTGTop8 files the commander(s) there (checked 2026-11-11
    on EDH, cEDH, EDHM and EDHP decks: 99 + 1, or 98 + 2 partners);
  - BETWEEN_GAMES (constructed formats): only a sideboard card that is
    one of the ten companions (_COMPANION_NAMES), one copy: a companion
    starts the game revealed outside the deck. The rest is dropped;
  - CARD_POOL (Limited): nothing; the "sideboard" is the unplayed rest
    of the pool, so a companion there is not the deck's.

CARD NAMES: MTGO writes split cards as "Fire/Ice"; the binder names
them "Fire // Ice", so a lone "/" becomes " // ". Two-faced cards are
written by their front face, which the matcher's name lookup accepts.
No card ids: every entry matches by name.
"""

import re
from enum import Enum

from src.data_refinement.deck_box.mtg_crawls.crawled_deck import (
    CrawledDeck,
    DeckSkip,
    MtgCardRef,
    MtgDeckEntry,
)
from src.data_retrieval.mtgtop8.formats import MtgTop8Format
from src.schema.json_checks import optional_str, require_int, require_str
from src.schema.json_types import JsonObject

_SIDEBOARD_DIVIDER = "Sideboard"
_CARD_LINE_PATTERN = re.compile(r"^(\d+) +(\S.*)$")


class SideboardRole(Enum):
    """What an MTGTop8 export's "Sideboard" section holds, by format."""

    COMMAND_ZONE = "command_zone"  # the commander(s): all kept
    BETWEEN_GAMES = "between_games"  # a real sideboard: companions kept
    CARD_POOL = "card_pool"  # Limited's unplayed cards: none kept


def _sideboard_role_of(deck_format: MtgTop8Format) -> SideboardRole:
    """What deck_format's "Sideboard" section holds.
    Inputs: deck_format. Output: SideboardRole. Side effects: none.
    Exceptions: none."""
    match deck_format:
        case (
            MtgTop8Format.DUEL_COMMANDER
            | MtgTop8Format.CEDH
            | MtgTop8Format.MTGO_COMMANDER
            | MtgTop8Format.COMMANDER_PRECONS
        ):
            return SideboardRole.COMMAND_ZONE
        case MtgTop8Format.LIMITED:
            return SideboardRole.CARD_POOL
        case _:
            return SideboardRole.BETWEEN_GAMES


# Every MtgTop8Format member's role
_SIDEBOARD_ROLES: dict[MtgTop8Format, SideboardRole] = {
    deck_format: _sideboard_role_of(deck_format) for deck_format in MtgTop8Format
}
_COMPANION_NAMES = frozenset(
    {
        "Gyruda, Doom of Depths",
        "Jegantha, the Wellspring",
        "Kaheera, the Orphanguard",
        "Keruga, the Macrosage",
        "Lurrus of the Dream-Den",
        "Lutri, the Spellchaser",
        "Obosh, the Preypiercer",
        "Umori, the Collector",
        "Yorion, Sky Nomad",
        "Zirda, the Dawnwaker",
    }
)


class MtgTop8DeckParser:
    """A CrawledDeckParser for MTGTop8 rows. Never skips a whole deck."""

    def parse(self, row: JsonObject) -> CrawledDeck | DeckSkip:
        """One MTGTop8 row -> its deck.

        Inputs: row (one crawl row, RAW SHAPE above).
        Output: CrawledDeck (only the cards that count as the deck).
        Side effects: none.
        Exceptions: ValueError naming the deck id on a malformed row or
            export line.

        Example:
            >>> MtgTop8DeckParser().parse(row).source_id
            '909096'
        """
        deck_id = row.get("deck_id")
        try:
            # Read the row's fields and split the export into sections
            source_id = str(require_int(row["deck_id"]))
            deck_format = MtgTop8Format(require_str(row["format"]))
            main, sideboard = _export_sections(require_str(row["mtgo"]))
            role = _SIDEBOARD_ROLES[deck_format]

            # The deck: main, plus the sideboard cards that start the game
            entries = main + tuple(
                kept
                for entry in sideboard
                if (kept := _sideboard_entry_in_play(entry, role)) is not None
            )
            return CrawledDeck(
                source_id, optional_str(row.get("deck_name")) or None, entries
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed MTGTop8 deck {deck_id!r}: {error}") from error


def _export_sections(
    export: str,
) -> tuple[tuple[MtgDeckEntry, ...], tuple[MtgDeckEntry, ...]]:
    """The export's (main, sideboard) entries: card lines before the
    divider, and after it; blank lines skipped; lines with a count
    below 1 dropped (as the other sites' parsers do).
    Inputs: export. Output: (main, sideboard). Side effects: none.
    Exceptions: ValueError on a line that is neither "<count> <name>"
        nor the divider, or a second divider."""
    main: list[MtgDeckEntry] = []
    sideboard: list[MtgDeckEntry] = []
    section = main

    # Card lines go to the current section; the divider switches it
    for raw_line in export.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == _SIDEBOARD_DIVIDER:
            if section is sideboard:
                raise ValueError("a second Sideboard divider")
            section = sideboard
            continue
        entry = _entry_from_line(line)
        if entry is not None:
            section.append(entry)
    return tuple(main), tuple(sideboard)


def _entry_from_line(line: str) -> MtgDeckEntry | None:
    """One "<count> <name>" line -> its entry (name through
    _binder_card_name), or None for a count below 1.
    Inputs: line (stripped, non-empty, not the divider).
    Output: MtgDeckEntry or None. Side effects: none.
    Exceptions: ValueError if it isn't "<count> <name>"."""
    match = _CARD_LINE_PATTERN.match(line)
    if match is None:
        raise ValueError(f"not a card line: {line!r}")
    count = int(match.group(1))
    if count < 1:
        return None
    return MtgDeckEntry(count, _name_only_card_ref(_binder_card_name(match.group(2))))


def _binder_card_name(export_name: str) -> str:
    """An MTGO card name as the binder writes it: "A/B" -> "A // B";
    other names unchanged.
    Inputs: export_name. Output: str. Side effects: none.
    Exceptions: none.

    Example:
        >>> _binder_card_name("Fire/Ice")
        'Fire // Ice'
    """
    if " // " in export_name:
        return export_name
    return " // ".join(face.strip() for face in export_name.split("/"))


def _sideboard_entry_in_play(
    entry: MtgDeckEntry, role: SideboardRole
) -> MtgDeckEntry | None:
    """WHAT COUNTS AS THE DECK (module docstring) for one sideboard
    entry: COMMAND_ZONE -> the entry; BETWEEN_GAMES -> one copy if it
    is a companion, else None; CARD_POOL -> None.
    Inputs: entry, role (the deck's format's SideboardRole).
    Output: MtgDeckEntry or None. Side effects: none.
    Exceptions: none."""
    match role:
        case SideboardRole.COMMAND_ZONE:
            return entry
        case SideboardRole.BETWEEN_GAMES if entry.card.name in _COMPANION_NAMES:
            return MtgDeckEntry(1, entry.card)
        case _:
            return None


def _name_only_card_ref(name: str) -> MtgCardRef:
    """The MtgCardRef for a name-only entry (no oracle or printing id).
    Inputs: name (binder spelling). Output: MtgCardRef.
    Side effects: none. Exceptions: none."""
    return MtgCardRef(name=name, oracle_id=None, printing_id=None)
