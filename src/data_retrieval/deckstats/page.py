"""Reading a Deckstats page's embedded Inertia.js state: the JSON in
<script data-page="app" type="application/json">, {"component",
"props", "url", "version", "sharedProps"}.

This is the "unwrap" step for an HTML source: the page JSON is the
data, so nothing else of the HTML is read.
"""

import json
import re
from typing import Any

_PAGE_SCRIPT_PATTERN = re.compile(
    r'<script[^>]*\bdata-page="app"[^>]*>(.*?)</script>', re.DOTALL
)


def parse_inertia_page(html: str) -> dict[str, Any]:
    """The page JSON embedded in html.

    Inputs: html (a Deckstats page). Output: the parsed page object.
    Side effects: none.
    Exceptions: ValueError if html has no data-page script, or its JSON
        does not parse or is not an object (a Cloudflare challenge or
        maintenance page: read_outcome treats it as transient).

    Example:
        >>> parse_inertia_page('<script data-page="app" type="application/json">'
        ...     '{"component": "DeckPrivate", "props": {}}</script>')["component"]
        'DeckPrivate'
    """
    match = _PAGE_SCRIPT_PATTERN.search(html)
    if match is None:
        raise ValueError("no Inertia page JSON (data-page script) in the page")
    page = json.loads(match.group(1))  # JSONDecodeError is a ValueError
    if not isinstance(page, dict):
        raise ValueError(f"page JSON is {type(page).__name__}, not an object")
    return page
