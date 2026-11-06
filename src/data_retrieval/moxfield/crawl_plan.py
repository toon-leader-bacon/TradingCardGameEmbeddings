"""Moxfield's walk: every deck its sitemap lists, once, in a fixed
seeded shuffle, so any prefix of the crawl is a uniform sample of the
sitemap.

Phase 1 lands each deck sitemap as one "publicId\\tlastmod" file
(sitemaps/deck-<nnnnn>.tsv) and merges them into crawl_plan.tsv, the
same line format in walk order. MoxfieldPlanFile reads the plan lazily
as a resumable_crawl.CrawlPlan with no skipped positions.
"""

import hashlib
import random
import re
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from src.data_retrieval.resumable_crawl import write_text_atomically

PLAN_NAME = "crawl_plan.tsv"
_PLAN_SEED = 20261105
_DECK_SITEMAP_PATTERN = re.compile(r"<loc>\s*([^<]*/sitemap-deck-(\d+)\.xml)\s*</loc>")
# One <url> entry: a deck URL (not "/primer") and its optional lastmod
_SITEMAP_DECK_PATTERN = re.compile(
    r"<loc>\s*[^<]*/decks/([A-Za-z0-9_-]{22})\s*</loc>\s*"
    r"(?:<lastmod>\s*([^<]+?)\s*</lastmod>)?"
)
_PUBLIC_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22}$")


@dataclass(frozen=True)
class SitemapDeck:
    """One deck URL from a sitemap."""

    public_id: str
    lastmod: str  # the sitemap's date, "" when absent

    def to_line(self) -> str:
        """This deck as one "publicId\\tlastmod" line, no newline.
        Inputs: none. Output: str. Side effects: none. Exceptions: none."""
        return f"{self.public_id}\t{self.lastmod}"

    @classmethod
    def from_line(cls, line: str) -> "SitemapDeck":
        """Parse one "publicId\\tlastmod" line (a trailing newline is
        ignored).

        Inputs: line. Output: SitemapDeck. Side effects: none.
        Exceptions: ValueError if it isn't two tab-separated fields with
            a 22-character public id.

        Example:
            >>> SitemapDeck.from_line("KtNrenwexES3cLg2NDMEnw\\t2022-12-22\\n")
            SitemapDeck(public_id='KtNrenwexES3cLg2NDMEnw', lastmod='2022-12-22')
        """
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 2 or not _PUBLIC_ID_PATTERN.match(fields[0]):
            raise ValueError(f"malformed plan line {line!r}")
        return cls(fields[0], fields[1])


class MoxfieldPlanFile:
    """crawl_plan.tsv: one "publicId\\tlastmod" line per deck, in walk
    order. Read lazily: ~5.7M lines (~190 MB) are never all in memory
    during phase 2."""

    def __init__(self, path: Path) -> None:
        """Inputs: path. Output: none. Side effects: none.
        Exceptions: none."""
        self.path = path

    @cached_property
    def fingerprint(self) -> str:
        """SHA-256 (hex) of the file. Inputs: none. Output: str.
        Side effects: reads it once. Exceptions: OSError.

        Example:
            >>> len(MoxfieldPlanFile(Path("crawl_plan.tsv")).fingerprint)
            64
        """
        with open(self.path, "rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    @cached_property
    def length(self) -> int:
        """Lines in the file. Inputs: none. Output: int.
        Side effects: reads it once. Exceptions: OSError.

        Example:
            >>> MoxfieldPlanFile(Path("crawl_plan.tsv")).length
            5712345
        """
        with open(self.path, "rb") as file:
            return sum(1 for _ in file)

    def ids_from(self, position: int) -> Iterator[tuple[int, str]]:
        """(position, publicId) for every line from position on.

        Inputs: position (>= 0). Output: an iterator over the pairs.
        Side effects: reads the file lazily, skipping the first position
            lines. A walk reads it twice per session (to start, and to
            check for an end of only skips): ~1 s each at the plan's end.
        Exceptions: OSError; ValueError for a malformed line.

        Example:
            >>> next(MoxfieldPlanFile(Path("crawl_plan.tsv")).ids_from(0))
            (0, 'KtNrenwexES3cLg2NDMEnw')
        """
        with open(self.path, encoding="utf-8", newline="\n") as file:
            for walk_position, line in enumerate(
                islice(file, position, None), start=position
            ):
                yield walk_position, SitemapDeck.from_line(line).public_id


def parse_deck_sitemap_urls(index_xml: str) -> dict[int, str]:
    """The deck sitemaps a sitemap index lists, number -> URL, in number
    order.

    Inputs: index_xml (sitemap.xml's text). Output: dict[int, str].
    Side effects: none. Exceptions: ValueError if none.

    Example:
        >>> parse_deck_sitemap_urls("<loc>https://moxfield.com/sitemap-deck-2.xml</loc>")
        {2: 'https://moxfield.com/sitemap-deck-2.xml'}
    """
    urls = {
        int(number): url for url, number in _DECK_SITEMAP_PATTERN.findall(index_xml)
    }
    if not urls:
        raise ValueError("Moxfield's sitemap index lists no deck sitemaps")
    return dict(sorted(urls.items()))


def parse_sitemap_decks(xml_text: str) -> list[SitemapDeck]:
    """The deck URLs in one sitemap (no "/primer"), in file order, without
    repeats (the first wins).

    Inputs: xml_text. Output: list[SitemapDeck]. Side effects: none.
    Exceptions: none.

    Example:
        >>> parse_sitemap_decks("<url><loc>https://moxfield.com/decks/"
        ...     "KtNrenwexES3cLg2NDMEnw</loc><lastmod>2022-12-22</lastmod></url>")
        [SitemapDeck(public_id='KtNrenwexES3cLg2NDMEnw', lastmod='2022-12-22')]
    """
    decks: dict[str, SitemapDeck] = {}
    for public_id, lastmod in _SITEMAP_DECK_PATTERN.findall(xml_text):
        decks.setdefault(public_id, SitemapDeck(public_id, lastmod))
    return list(decks.values())


def write_sitemap_decks(path: Path, decks: Iterable[SitemapDeck]) -> None:
    """Write decks, one "publicId\\tlastmod" line each, atomically: a
    landed sitemap, or the plan (decks in walk order).
    Inputs: path, decks. Output: none. Side effects: writes path.
    Exceptions: OSError.

    Example:
        >>> write_sitemap_decks(Path("deck-00001.tsv"), [SitemapDeck("x" * 22, "")])
    """
    write_text_atomically(path, "".join(deck.to_line() + "\n" for deck in decks))


def merge_and_shuffle(landed_paths: Iterable[Path]) -> list[SitemapDeck]:
    """Every deck in the landed sitemap files, once (the latest lastmod
    wins; ISO dates sort as text), sorted by public id and then shuffled
    with random.Random(_PLAN_SEED), so the plan depends only on the set
    of decks, not on the sitemaps' order.

    Inputs: landed_paths. Output: the decks in plan order.
    Side effects: reads the files.
    Exceptions: OSError; ValueError for a malformed line or no decks.

    Example:
        >>> len(merge_and_shuffle(sorted(Path("sitemaps").glob("*.tsv"))))
        5712345
    """
    latest: dict[str, SitemapDeck] = {}
    for path in landed_paths:
        with open(path, encoding="utf-8", newline="\n") as file:
            for line in file:
                deck = SitemapDeck.from_line(line)
                known = latest.get(deck.public_id)
                if known is None or deck.lastmod > known.lastmod:
                    latest[deck.public_id] = deck
    if not latest:
        raise ValueError("the landed Moxfield sitemaps hold no decks")

    result = [latest[public_id] for public_id in sorted(latest)]
    random.Random(_PLAN_SEED).shuffle(result)
    return result
