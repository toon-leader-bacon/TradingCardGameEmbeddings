"""Typed decklists parsed from Limitless standings rows.

The raw shape (see src/data_retrieval/limitless_play/downloader.py): a
standings row's "decklist" is {"pokemon": [...], "trainer": [...],
"energy": [...]}, each entry {"count", "set", "number", "name"}, or
null when the player submitted none. A multi-deck tournament instead
gives "decklists": {"1": {...}, "2": {...}}, with each deck's archetype
under the same key in "decks". The section a card sits in is not kept:
the card itself says whether it's a Pokemon, Trainer or Energy.

Other games name other sections (One Piece: "leader", "character",
"event", "stage"). Two shapes differ from Pokemon's:

- A single-card section ("leader", Star Wars: Unlimited's "base") is
  one entry object with no "count": it is one copy.
- "side" is the sideboard, not part of the deck a game starts with: it
  is dropped, as ygoprodeck_decks and mtgjson_decks drop sideboards.
"""

from dataclasses import dataclass

# Not part of the deck a game starts with
_SIDEBOARD_SECTION = "side"


@dataclass(frozen=True)
class LimitlessCardRef:
    """The fields that name one printing: a card matcher's cache key.

    set_code is Limitless's set code (PTCGL style, e.g. "PAR", "SVP";
    "OP13" for One Piece), number the collector number as printed (a
    string: "57", "TG12", "001").
    """

    set_code: str
    number: str
    name: str


@dataclass(frozen=True)
class LimitlessCardEntry:
    """One decklist line: count copies of one printing (a CountedCard)."""

    count: int
    card: LimitlessCardRef


@dataclass(frozen=True)
class LimitlessDecklist:
    """One player's submitted decklist in one tournament.

    source_id is "<tournament id>:<standings row index>", stable for a
    given raw file and free of player names.
    """

    source_id: str
    archetype_name: str | None  # Limitless's auto-assigned deck type
    entries: tuple[LimitlessCardEntry, ...]

    @property
    def card_count(self) -> int:
        """Total copies across entries.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return sum(entry.count for entry in self.entries)


def parse_standings_row(
    tournament_id: str, row_index: int, row: dict
) -> list[LimitlessDecklist]:
    """Every decklist one standings row carries: none, one
    ("decklist"), or several ("decklists", multi-deck tournaments).

    Inputs: tournament_id, row_index (the row's position in the
        tournament's standings), row (one raw standings row).
    Output: the parsed decklists; a multi-deck row's source ids gain
        ":<deck key>".
    Side effects: none.
    Exceptions: KeyError / TypeError / ValueError on a malformed entry.

    Example:
        >>> parse_standings_row("6adb37", 0, row)[0].card_count
        60
    """
    result: list[LimitlessDecklist] = []
    source_id = f"{tournament_id}:{row_index}"

    # A single-deck row
    if row.get("decklist"):
        result.append(
            LimitlessDecklist(
                source_id,
                _read_archetype_name(row.get("deck")),
                _parse_entries(row["decklist"]),
            )
        )

    # A multi-deck row: one archetype per deck key, in row["decks"]
    archetypes = row.get("decks") or {}
    for deck_key, raw_decklist in (row.get("decklists") or {}).items():
        if raw_decklist:
            result.append(
                LimitlessDecklist(
                    f"{source_id}:{deck_key}",
                    _read_archetype_name(archetypes.get(deck_key)),
                    _parse_entries(raw_decklist),
                )
            )
    return result


def _parse_entries(raw_decklist: dict) -> tuple[LimitlessCardEntry, ...]:
    """Every entry across a raw decklist's sections but the sideboard,
    in section order.

    Inputs: raw_decklist ({section: [entry, ...] or one entry}).
    Output: the parsed entries. Side effects: none.
    Exceptions: KeyError / ValueError / TypeError on a malformed entry.
    """
    return tuple(
        entry
        for section, raw_section in raw_decklist.items()
        if section != _SIDEBOARD_SECTION
        for entry in _parse_section(raw_section)
    )


def _parse_section(raw_section: list | dict) -> list[LimitlessCardEntry]:
    """A section's entries: each entry of a list section (each with its
    "count"), or a single-card section's one entry object as one copy.

    Inputs: raw_section. Output: the parsed entries.
    Side effects: none.
    Exceptions: TypeError if raw_section is neither a list nor a dict;
        KeyError / ValueError / TypeError on a malformed entry.
    """
    if isinstance(raw_section, dict):
        return [_parse_entry(raw_section, count=1)]
    if not isinstance(raw_section, list):
        raise TypeError(f"a decklist section is a list or an entry: {raw_section!r}")
    return [
        _parse_entry(raw_entry, count=int(raw_entry["count"]))
        for raw_entry in raw_section
    ]


def _parse_entry(raw_entry: dict, count: int) -> LimitlessCardEntry:
    """One raw entry, its copies given by the caller.

    Inputs: raw_entry ({"set", "number", "name", ...}), count.
    Output: LimitlessCardEntry. Side effects: none.
    Exceptions: KeyError / TypeError on a malformed entry.
    """
    return LimitlessCardEntry(
        count=count,
        card=LimitlessCardRef(
            set_code=str(raw_entry["set"]),
            number=str(raw_entry["number"]),
            name=str(raw_entry["name"]),
        ),
    )


def _read_archetype_name(raw_deck: dict | None) -> str | None:
    """The archetype's "name", or None when unassigned.

    Inputs: raw_deck ({"id", "name", "icons"} or None).
    Output: str | None. Side effects: none. Exceptions: none.
    """
    if not raw_deck:
        return None
    return raw_deck.get("name") or None
