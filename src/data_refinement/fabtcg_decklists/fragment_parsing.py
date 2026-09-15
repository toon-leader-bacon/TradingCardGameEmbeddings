"""Shared HTML-walking logic for one fabtcg.com decklist fragment.

Sole current consumer is src/data_refinement/deck_box/fabtcg_decklists/
extraction_stage.py. This module owns only the "how to walk a fragment
into (quantity, name) pairs" mechanics; what a caller does with each
pair (resolve it, and how to handle a miss) is entirely that caller's
own concern.

FRAGMENT SHAPE (confirmed live against data/tmp/list_view.html):
src/data_retrieval/fabtcg_decklists/ saves one prettified
`<section class="decklist-list-view block hidden">` fragment per deck.
Every group inside it — the combined "Hero / Weapon / Equipment" group,
and each of "Pitch 1"/"Pitch 2"/"Pitch 3" — is a uniformly-shaped
`<div class="list-view-container">`, so a single
`soup.select("div.list-view-container")` finds all of them regardless
of the pitch groups' extra `<section class="pitch-section">` wrapper —
no special-casing needed. Each `<li class="card-item group">` holds
`<div class="card-name"><span>1x</span> Card Name</div>` — quantity and
name are not separate elements, so extraction reads
`" ".join(name_div.get_text().split())` ("1x Card Name") and splits it
with _QUANTITY_AND_NAME_PATTERN. The split()/join() (rather than
get_text(" ", strip=True)) is deliberate: some real card-name divs wrap
their text across an internal newline as one text node — get_text's
separator only joins BETWEEN separate text nodes, so it would never
collapse an internal one, and a long name plus a trailing pitch-color
marker on its own line would otherwise fail to match
_QUANTITY_AND_NAME_PATTERN.

PITCH-COLOR MARKER: fabtcg.com appends a literal " (red)"/" (yel)"/
" (blu)" suffix to a card-name div's text whenever that deck includes
more than one pitch-printing of the same card name (confirmed live —
every occurrence across data/raw/fabtcg_decklists/decklists/*.html is
exactly one of those three strings; card-name divs for a name with only
one pitch-printing in the deck carry no suffix at all). This is
fabtcg.com UI disambiguation, not part of the card's actual name — the
card_binder's stored name for e.g. "Herald of Triumph (blu)" is plainly
"Herald of Triumph" — so _parse_quantity_and_name() strips it via
_TRAILING_PITCH_COLOR_PATTERN before returning the name. A caller doing
an exact-name card_lookup would otherwise miss on every card name that
carries this suffix.

MANGLED UNICODE ESCAPES: fabtcg.com's own page template drops the
backslash off any \\uXXXX JSON-style escape it embeds into a
card-name div's text, so e.g. the card actually named "Olé" (stored in
card_binder as "Olé") shows up literally as "Olu00e9" in the
fragment, and "Twelve Petal Kāṣāya" as "Twelve Petal
Ku0101u1e63u0101ya" — CONFIRMED LIVE by fetching
https://fabtcg.com/decklists/34eb2711-34f9-4f94-9058-2a8e1cd27160/
directly (curl, browser User-Agent) and finding the exact same
"Ku0101u1e63u0101ya" bytes already present on the wire — this is
fabtcg.com's own bug, not something src/data_retrieval/fabtcg_decklists/
introduces or could avoid by fetching differently. It's mechanically
reversible: _parse_quantity_and_name() restores every "u" + 4 lowercase
hex digits run via _MANGLED_UNICODE_ESCAPE_PATTERN, the same
reconstruction a leading backslash would have triggered. Deliberately
lowercase-hex-only (matching every confirmed occurrence) — real card
name text has no other plausible reason to contain "u" immediately
followed by 4 lowercase hex digits, so this never fires on a genuine
name.

FIRST ITERATION FLATTENS GROUPS: which group each card came from is
discarded by iter_card_quantities_and_names() — every caller today only
wants a flat per-deck card list. See fabtcg_decklists/TODO.md for the
deferred structured/grouped follow-up and why that's a deliberate
simplification, not an oversight.
"""

import re
from typing import Iterator

from bs4 import BeautifulSoup

_LIST_VIEW_CONTAINER_SELECTOR = "div.list-view-container"
_CARD_ITEM_SELECTOR = "li.card-item"
_CARD_NAME_SELECTOR = "div.card-name"
_QUANTITY_AND_NAME_PATTERN = re.compile(r"^(\d+)x\s+(.+)$")
_TRAILING_PITCH_COLOR_PATTERN = re.compile(r"\s+\((?:red|yel|blu)\)$")
_MANGLED_UNICODE_ESCAPE_PATTERN = re.compile(r"u([0-9a-f]{4})")


def iter_card_quantities_and_names(fragment_html: str) -> Iterator[tuple[int, str]]:
    """Walk every card group in one deck fragment, yielding each
    card-item's (quantity, name), in document order.

    See this module's docstring's FRAGMENT SHAPE and FIRST ITERATION
    FLATTENS GROUPS sections for exactly what's walked and what's
    deliberately discarded (group membership). Purely mechanical — this
    function does no card resolution; that's every caller's own
    concern.

    Inputs:
        fragment_html: one deck's saved decklist fragment.
    Output: one (quantity, name) tuple per card-item element found in
        fragment_html, in document order. Empty if fragment_html
        contains zero card-item elements — callers decide for
        themselves whether that's an error.
    Side effects: none.
    Exceptions: raises ValueError if a card-item's name text doesn't
        match _QUANTITY_AND_NAME_PATTERN, or if a card-item has no
        card-name child element.

    Example:
        >>> list(iter_card_quantities_and_names(fragment_html))
        [(1, 'Dorinthea Ironsong'), (3, 'Command and Conquer')]
    """
    soup = BeautifulSoup(fragment_html, "html.parser")

    # Walk every group container (Hero/Weapon/Equipment, and each
    # Pitch N) uniformly — see module docstring's FRAGMENT SHAPE.
    for container in soup.select(_LIST_VIEW_CONTAINER_SELECTOR):
        for card_item in container.select(_CARD_ITEM_SELECTOR):
            name_div = card_item.select_one(_CARD_NAME_SELECTOR)
            if name_div is None:
                raise ValueError(
                    f"iter_card_quantities_and_names: a {_CARD_ITEM_SELECTOR!r} "
                    f"element has no {_CARD_NAME_SELECTOR!r} child — "
                    "fabtcg.com's markup may have changed"
                )

            # " ".join(...split()) rather than get_text(" ",
            # strip=True): some real card-name divs wrap their text
            # across an internal newline (e.g. a long name plus a
            # trailing "(red)" pitch marker on its own line) as ONE
            # text node — get_text's separator only joins BETWEEN
            # separate text nodes, so it never collapses that internal
            # newline, and _QUANTITY_AND_NAME_PATTERN (whose `.`
            # doesn't match newlines) would otherwise raise on a
            # perfectly valid card. split()/join() normalizes every
            # run of whitespace uniformly.
            yield _parse_quantity_and_name(" ".join(name_div.get_text().split()))


def _parse_quantity_and_name(name_div_text: str) -> tuple[int, str]:
    """Split one card-name div's flattened text into its quantity and
    card name, with any trailing pitch-color marker removed and any
    mangled unicode escape repaired.

    Private helper — single consumer is
    iter_card_quantities_and_names(). E.g. "1x Dorinthea Ironsong" ->
    (1, "Dorinthea Ironsong"); "3x Herald of Triumph (blu)" -> (3,
    "Herald of Triumph") — see module docstring's PITCH-COLOR MARKER
    section for why that suffix is stripped rather than kept; "1x
    Olu00e9" -> (1, "Olé") — see module docstring's MANGLED UNICODE
    ESCAPES section.

    Inputs:
        name_div_text: a `div.card-name` element's text, with every run
            of whitespace (including any internal newline) already
            collapsed to a single space.
    Output: (quantity, name), name never carrying a trailing " (red)"/
        " (yel)"/" (blu)" marker, and with every mangled "u" + 4
        lowercase hex digit run restored to its real character.
    Side effects: none.
    Exceptions: raises ValueError if name_div_text doesn't match
        _QUANTITY_AND_NAME_PATTERN.
    """
    match = _QUANTITY_AND_NAME_PATTERN.match(name_div_text)
    if match is None:
        raise ValueError(
            f"_parse_quantity_and_name: {name_div_text!r} does not match "
            f"the expected {_QUANTITY_AND_NAME_PATTERN.pattern!r} shape"
        )

    quantity, name = match.groups()
    name = _TRAILING_PITCH_COLOR_PATTERN.sub("", name)
    name = _MANGLED_UNICODE_ESCAPE_PATTERN.sub(
        lambda escape: chr(int(escape.group(1), 16)), name
    )
    return int(quantity), name
