"""MTGTop8's MTGO-format deck export (/mtgo?d=<id>): checked, not
parsed. The text is landed as the site writes it; splitting it into
cards is data_refinement's job.

SHAPE (probed 2026-11-11): text/plain, ISO-8859-1, CRLF lines:
    4 Goblin Bushwhacker
    ...
    Sideboard
    2 Smash to Smithereens
A deck id with no deck answers 200 with an empty body.
"""

import re

# "<count> <name>", or the "Sideboard" divider; blank lines allowed
_EXPORT_LINE_PATTERN = re.compile(r"^(\d+ \S.*|Sideboard|)$")


def read_deck_export(text: str) -> str | None:
    """The export text if it is a deck export, None for the empty answer
    to a missing deck.

    Inputs: text (the response body, decoded).
    Output: text unchanged, or None if it is empty or whitespace.
    Side effects: none.
    Exceptions: ValueError if any line is neither a card line nor the
        divider (an HTML error or maintenance page: read_outcome treats
        it as transient).

    Example:
        >>> read_deck_export("4 Lightning Bolt\\r\\nSideboard\\r\\n1 Pyroblast\\r\\n")
        '4 Lightning Bolt\\r\\nSideboard\\r\\n1 Pyroblast\\r\\n'
        >>> read_deck_export("") is None
        True
    """
    if not text.strip():
        return None
    for line in text.splitlines():
        if not _EXPORT_LINE_PATTERN.match(line.strip()):
            raise ValueError(f"not an MTGO deck export: line {line[:80]!r}")
    return text
