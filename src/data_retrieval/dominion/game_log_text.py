"""Turns dominion.games' game-log HTML into indented plain text.

Split out of dominion_parser.py so the text conversion can be tested
without Selenium or a browser.
"""

import re

# One match per piece of the log that matters: a class attribute (marks
# a new turn), an inline padding style (sets the line's indent), or a
# span's text.
_LOG_PIECE = re.compile(
    r'(class="[^"]*"|style="padding-left: [^%]*|<span>[^<]*</span>)'
)
_SPAN_OPEN = "<span>"
_SPAN_CLOSE = "</span>"
_INDENT_WIDTH = 4


def game_log_text(log_html: str) -> str:
    """Convert the innerHTML of the game-log element into plain text.

    Inputs:
        log_html: the game-log element's innerHTML.
    Output: the log as text. A new turn starts after a blank line; each
        line is indented 4 spaces per distinct padding level, numbered
        in the order the levels first appear (the first level seen is
        0).
    Side effects: none.
    Exceptions: none.

    Example:
        >>> game_log_text('<div class="new-turn-line"></div>'
        ...     '<div style="padding-left: 0%"><span>Turn 1</span></div>')
        '\\n\\n\\nTurn 1'
    """
    indent_levels = ["0"]
    pieces: list[str] = []

    for match in _LOG_PIECE.findall(log_html.replace("\n", "")):
        if match.startswith("class"):
            if "new-turn-line" in match:
                pieces.append("\n\n")
        elif match.startswith("style"):
            indent = match.split()[-1]
            if indent not in indent_levels:
                indent_levels.append(indent)
            pieces.append("\n" + " " * _INDENT_WIDTH * indent_levels.index(indent))
        else:
            pieces.append(match[len(_SPAN_OPEN) : -len(_SPAN_CLOSE)])

    return "".join(pieces)
