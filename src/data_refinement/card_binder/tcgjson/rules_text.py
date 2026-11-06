"""Cleaning TCGplayer's rules-text HTML, shared by every tcgjson game.

TCGplayer's text is typed per product: <strong>/<em>/<br> markup, a few
<span style=...> and typo tags (<stong>, <strrong>, an unclosed
"<strong[I Wreck Things]"), NBSPs and curly quotes. Printings of one card
must clean to the same string wherever only this markup differs, so the
vote in printing_groups.py sees one value. Brackets are game text ([Exert],
ability names) and are kept.

TCGplayer also writes store notes into the text, in a red span or on a
line of their own ("Disclaimer: This card was reprinted ...", "This
product comes in Three (3) serializations", "Official Tournament Stores
must review ..."). They are not card text and are dropped: a red span,
or a <br>-separated line, whose text opens like a note. Only such a
span goes, because a red span is also how a printing highlights
errata'd card text ("Then, <span style="color:red">you may</span> play
..."); and the openings are narrow ("This card has been", not "This
card"), because card text opens lines with "This card ..." too.
"""

import re

# Every tag, closed or not, including typo names: <br>, </strong>, <stong>, <strong[
_TAG = re.compile(r"</?[A-Za-z][^<>\[]*>?")
_STRAIGHT_PUNCTUATION = str.maketrans(
    {
        " ": " ",  # no-break space
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)
_WHITESPACE_RUN = re.compile(r"\s+")
# The one red-span form seen; no span nested inside one was seen
_RED_SPAN = re.compile(r'<span style="color:\s*red;?">(.*?)</span>', re.DOTALL)
_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
# How every store note seen opens (weekly-20260831), each with a game
# it was seen in; matched case-insensitively, after an optional "("
_STORE_NOTE_OPENINGS = (
    "disclaimer",  # every game but Union Arena
    "please note",  # Yu-Gi-Oh
    "official tournament stores",  # Yu-Gi-Oh
    "this product",  # Star Wars: Unlimited
    "this promotional",  # Yu-Gi-Oh
    "this double-sided",  # Riftbound, Yu-Gi-Oh
    "this european exclusive",  # Yu-Gi-Oh
    "this the falsebound",  # Yu-Gi-Oh: TCGplayer's own wording, not a typo
    "this card's text has",  # Lorcana
    "this card was reprinted",  # Yu-Gi-Oh
    "this card was re-printed",  # One Piece
    "this card was officially",  # Lorcana, Yu-Gi-Oh
    "this card has been",  # One Piece
    "this card features a",  # Yu-Gi-Oh
    "this card is not tournament",  # Yu-Gi-Oh
    "this card is not legal",  # Yu-Gi-Oh
    "this card is considered oversized",  # Riftbound
    "this card is considered not tournament",  # One Piece
)
_STORE_NOTE_OPENING = re.compile(
    r"\(?(" + "|".join(re.escape(o) for o in _STORE_NOTE_OPENINGS) + ")",
    re.IGNORECASE,
)
# A space a removed tag left before closing punctuation, or after an opener
_SPACE_BEFORE_CLOSER = re.compile(r" (?=[,.;:!?)])")
_SPACE_AFTER_OPENER = re.compile(r"(?<=\() ")


def clean_rules_text(html: str) -> str:
    """html without its markup: every tag becomes a space (so
    "[Dragon Fire]</strong>When" keeps two words), NBSP becomes a space,
    curly quotes become straight, whitespace runs collapse to one space,
    and a space before , . ; : ! ? ) or after ( is removed (so
    "<em>(Evasive</em>)" and "(Evasive)" clean alike). Store notes, in a
    red span or on a line of their own, are dropped first (see the
    module docstring).

    Inputs: html (a product's rulesText).
    Output: the cleaned text, stripped ("" if nothing is left).
    Side effects: none.
    Exceptions: none.

    Example:
        >>> clean_rules_text("<strong>[Freeze]</strong>[Exert] — Exert it.<br>")
        '[Freeze] [Exert] — Exert it.'
    """
    text = _RED_SPAN.sub(_drop_red_store_note, html)
    lines = [line for line in _LINE_BREAK.split(text) if not _is_store_note(line)]
    text = _TAG.sub(" ", " ".join(lines)).translate(_STRAIGHT_PUNCTUATION)
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _SPACE_BEFORE_CLOSER.sub("", text)
    return _SPACE_AFTER_OPENER.sub("", text).strip()


def _drop_red_store_note(match: re.Match[str]) -> str:
    """A red span's replacement: a space if it is a store note, else its
    inner HTML (cleaned with the rest).
    Inputs: match (of _RED_SPAN). Output: str.
    Side effects: none. Exceptions: none."""
    inner = match.group(1)
    return " " if _is_store_note(inner) else inner


def _is_store_note(html: str) -> bool:
    """Whether html's text, tags removed, opens like a store note.
    Inputs: html (a span's inner HTML, or one line). Output: bool.
    Side effects: none. Exceptions: none."""
    return bool(_STORE_NOTE_OPENING.match(_TAG.sub("", html).strip()))
