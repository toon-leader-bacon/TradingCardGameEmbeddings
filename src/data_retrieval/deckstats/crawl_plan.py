"""Deckstats' walk: the sitemap's deck paths ("<id>-<slug>") first, then
every id in [1, the highest sitemap id] in a fixed spread order (bare
"<id>"), skipping sitemap ids. A resumable_crawl.SitemapThenSpreadPlan
of str path tails, persisted as crawl_plan.json ({"sitemap_paths":
[...], "spread": {...}}).

Both item forms are just what follows /decks/ in a URL; the downloader
fetches them the same way (a slug that is stale or absent costs one
paced redirect).
"""

import re

from src.data_retrieval.download_utils import require_str
from src.data_retrieval.resumable_crawl import PlanItems

PLAN_NAME = "crawl_plan.json"
# Deck sitemaps only: qualified, evergreen, fresh (not the Lorcana one)
_DECK_SITEMAP_PATTERN = re.compile(
    r"<loc>\s*([^<]*/sitemaps/sitemap-decks-[a-z]+-\d+\.xml)\s*</loc>"
)
# A deck URL with no query or fragment (robots.txt disallows /*?)
_SITEMAP_DECK_PATTERN = re.compile(r"<loc>\s*[^<]*/decks/(\d+-[^<\s/?#]*)\s*</loc>")
# "<id>", "<id>-<slug>", optionally after "/decks/"
_DECK_PATH_PATTERN = re.compile(r"^(?:/decks/)?(\d+)(?:-[^/?#]*)?$")


def deck_id_of_path(deck_path: str) -> int:
    """The deck id a Deckstats deck path holds: "<id>", "<id>-<slug>",
    or either after "/decks/" (a page's "url").

    Inputs: deck_path. Output: int. Side effects: none.
    Exceptions: ValueError if it is none of those forms.

    Example:
        >>> deck_id_of_path("325-Burn-Mana-Burn"), deck_id_of_path("/decks/1705661")
        (325, 1705661)
    """
    match = _DECK_PATH_PATTERN.match(deck_path)
    if match is None:
        raise ValueError(f"not a Deckstats deck path: {deck_path!r}")
    return int(match.group(1))


# Items are deck path tails; a spread id is its own bare path
DECKSTATS_ITEMS = PlanItems[str](
    id_of_item=deck_id_of_path,
    item_of_id=str,
    read_json_item=require_str,
    sitemap_key="sitemap_paths",
)


def parse_deck_sitemap_urls(index_xml: str) -> list[str]:
    """The deck sitemaps (sitemap-decks-<kind>-<nn>.xml) a sitemap index
    lists, in listing order, without repeats.

    Inputs: index_xml. Output: list of URLs. Side effects: none.
    Exceptions: ValueError if none.

    Example:
        >>> parse_deck_sitemap_urls("<loc>https://deckstats.net/sitemaps/"
        ...     "sitemap-decks-fresh-00.xml</loc>")
        ['https://deckstats.net/sitemaps/sitemap-decks-fresh-00.xml']
    """
    urls = list(dict.fromkeys(_DECK_SITEMAP_PATTERN.findall(index_xml)))
    if not urls:
        raise ValueError("Deckstats' sitemap index lists no deck sitemaps")
    return urls


def parse_sitemap_deck_paths(xml_text: str) -> list[str]:
    """The "<id>-<slug>" path tail of every deck URL in one sitemap, in
    order.

    Inputs: xml_text. Output: list of paths. Side effects: none.
    Exceptions: none.

    Example:
        >>> parse_sitemap_deck_paths(
        ...     "<loc>https://deckstats.net/decks/325-Burn-Mana-Burn</loc>")
        ['325-Burn-Mana-Burn']
    """
    return _SITEMAP_DECK_PATTERN.findall(xml_text)


def keep_first_path_per_id(deck_paths: list[str]) -> tuple[str, ...]:
    """deck_paths in order, keeping each deck id's first path (a deck can
    sit in two sitemaps, or under two slugs).

    Inputs: deck_paths. Output: tuple of paths. Side effects: none.
    Exceptions: ValueError for a path with no deck id.

    Example:
        >>> keep_first_path_per_id(["2-Burn", "3-Elves", "2-Burn-v2"])
        ('2-Burn', '3-Elves')
    """
    first_paths: dict[int, str] = {}
    for deck_path in deck_paths:
        first_paths.setdefault(deck_id_of_path(deck_path), deck_path)
    return tuple(first_paths.values())
