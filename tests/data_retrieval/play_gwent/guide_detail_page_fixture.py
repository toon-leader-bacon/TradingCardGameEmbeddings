"""Real (trimmed) guide detail page HTML, for testing
PlayGwentDownloader._extract_guide_payload without a live network call.

The data-state payload here is a trimmed subset of a real response
captured from https://www.playgwent.com/en/decks/guides/407608 (fields
unrelated to what _extract_guide_payload cares about — images,
tooltips, fluff text, the full card list — were dropped to keep this
fixture small; the id/name/leader values that remain are the real
ones observed, not fabricated).
"""

import json

_REAL_GUIDE_PAYLOAD = {
    "guide": {
        "id": 407608,
        "name": "💰Дани Борсоди | OTB Borsodi🤝",
        "author": "MetallicDanny",
        "faction": {"short": "syn", "slug": "syndicate"},
        "leaderId": 202328,
        "leader": {
            "id": 202328,
            "name": "Off the Books",
            "localizedName": "Off the Books",
        },
        "deck": {
            "id": 2018264,
            "srcCardTemplates": [202328, 202305, 202379],
        },
        "cards": [
            {"id": 202305, "name": "Tiger's Eye"},
            {"id": 202379, "name": "Novigrad"},
        ],
    }
}

_DATA_STATE_JSON = json.dumps(_REAL_GUIDE_PAYLOAD)
_ESCAPED_DATA_STATE = (
    _DATA_STATE_JSON.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&#x27;")
)

REAL_GUIDE_ID = 407608
REAL_GUIDE_NAME = _REAL_GUIDE_PAYLOAD["guide"]["name"]
REAL_GUIDE_CARDS = _REAL_GUIDE_PAYLOAD["guide"]["cards"]

# A decoy <style> block containing dozens of unquoted `data-state=...`
# CSS attribute selectors (tippy.js's own stylesheet, inlined into the
# page when a browser does a "complete webpage" save) — a real
# example of the noise _extract_guide_payload must not be fooled by.
# Confirmed against real saved pages: "data-state" appears as a raw
# substring 52 times per page, of which only one (the real payload) is
# ever a quoted HTML attribute — but a naive substring/regex search is
# still fragile against this, which is exactly why extraction is
# scoped to the specific root div rather than searched for anywhere in
# the page text.
_DECOY_TIPPY_STYLE_BLOCK = """
<style>
.tippy-popper[x-placement^=top] .tippy-backdrop[data-state=visible] {
    transform: scale(1) translate(-50%, -55%)
}
.tippy-tooltip[data-state=hidden] { visibility: hidden }
</style>
"""

# A decoy element carrying its own *quoted* data-state attribute,
# unrelated to the guide payload — e.g. a UI widget's open/closed
# state. Not observed in a real captured page, but exactly the
# scenario a plain "first quoted data-state in the document" search
# would get wrong; div#root's own data-state must still win because
# lookup is scoped to it specifically.
_DECOY_QUOTED_ATTRIBUTE_ELEMENT = '<div data-state="open"></div>'

# Double-quoted data-state attribute on div#root — the form seen in a
# live, unauthenticated curl against the real site. Wrapped in the
# real div.wrapper > div.content > div#root structure, plus the two
# decoys above, both of which a naive regex/substring search could
# latch onto instead of the real element.
DOUBLE_QUOTED_GUIDE_PAGE_HTML = f"""
<!DOCTYPE html><html><head><title>GWENT: The Witcher Card Game</title>
{_DECOY_TIPPY_STYLE_BLOCK}
</head>
<body>
{_DECOY_QUOTED_ATTRIBUTE_ELEMENT}
<div class="wrapper"><div class="content">
<div id="root" data-url-base="/en/decks" data-state="{_ESCAPED_DATA_STATE}" data-translations="{{}}"></div>
</div></div></body></html>
"""

# Single-quoted data-state attribute — the form seen when saving the
# page from a browser (Chrome's "Save Page As", complete webpage).
SINGLE_QUOTED_GUIDE_PAGE_HTML = f"""
<!DOCTYPE html><html><head><title>GWENT: The Witcher Card Game</title>
{_DECOY_TIPPY_STYLE_BLOCK}
</head>
<body>
{_DECOY_QUOTED_ATTRIBUTE_ELEMENT}
<div class="wrapper"><div class="content">
<div id="root" data-url-base="/en/decks" data-state='{_ESCAPED_DATA_STATE}' data-translations="{{}}"></div>
</div></div></body></html>
"""

GUIDE_PAGE_HTML_WITHOUT_DATA_STATE = """
<!DOCTYPE html><html><head><title>GWENT: The Witcher Card Game</title></head>
<body><div class="wrapper"><div class="content"><div id="root"></div></div></div></body></html>
"""

GUIDE_PAGE_HTML_WITH_NO_ROOT_DIV = """
<!DOCTYPE html><html><body><div class="wrapper"><div class="content">
<div id="not-root"></div>
</div></div></body></html>
"""

GUIDE_PAGE_HTML_WITH_NO_GUIDE_KEY = """
<div class="wrapper"><div class="content">
<div id="root" data-state="{&quot;somethingElse&quot;:1}"></div>
</div></div>
"""
