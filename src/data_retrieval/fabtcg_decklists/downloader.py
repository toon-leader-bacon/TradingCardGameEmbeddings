"""Downloads Flesh and Blood decklist HTML from fabtcg.com, in two
phases.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job) — this class's only transformation is isolating
and pretty-printing the one HTML fragment a deck page already embeds
its decklist in, not reshaping its contents. See
plans/fabtcg_decklists.md for this feature's confirmed-live scope.

Distinct from cardvault_fabtcg (existing source, a different fabtcg.com
subdomain, pulls a static card CSV) — no overlap.

Two phases, kept as two separate public methods rather than composed
into one fetch() — same rationale as the sibling downloaders (they're
expected to be run and re-run as separate steps against a live site):

    1. phase_1() — fabtcg.com publishes its decklists across several
       numbered sitemaps (decklist-sitemap.xml, decklist-sitemap2.xml,
       ...), themselves listed inside a top-level sitemap index
       (/sitemap_index.xml, alongside many unrelated sitemaps for
       posts/products/etc — CONFIRMED LIVE 2026-09-06). This method
       fetches the index, finds every decklist sub-sitemap, fetches
       each of those, and extracts every real deck page URL
       ("https://fabtcg.com/decklists/<slug>/") — excluding a bare
       "/decklists/" entry with no slug and locale-prefixed variants
       ("/fr/decklists/...", "/ja/decklists/...", both CONFIRMED LIVE
       present in the real sitemaps) — writing the full URLs (not just
       slugs; phase_2() needs the whole URL, there's no separate
       id-to-URL template the way the JSON-API sources have one) to
       deck_urls.txt.

    2. phase_2() — reads deck_urls.txt and, for every URL whose deck
       page hasn't already been saved to disk, fetches that page and
       extracts+prettifies its
       "<section class="decklist-list-view block hidden">" fragment
       (the same fragment for every deck page; contains the deck's
       Hero/Weapon/Equipment and Pitch 1/2/3 groups — see
       plans/fabtcg_decklists.md), saving it to
       decklists/<slug>.html. Resumability here is "does this deck's
       output file already exist" rather than a shared-JSONL
       manifest — see this module's docstring further down for why
       read_manifest()/append_with_manifest() (used by the sibling
       JSONL-based downloaders) don't fit this one-file-per-deck shape.

CONFIRMED LIVE (2026-09-06): fabtcg.com's WAF returns 403 for the
default python-requests/curl User-Agent on every endpoint tested here
(the sitemap index, a decklist sub-sitemap, and a deck detail page
alike) but 200s for an ordinary browser-style User-Agent. No existing
source in this container needed this, so download_to_string() and
download_to_file() (src/data_retrieval/download_utils.py) each gain a
new optional `headers` param for this source to pass a constant
browser-like User-Agent through — backwards compatible, since every
other existing caller keeps passing nothing and gets identical
behavior to today.
"""

import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup
from tqdm import tqdm

from src.data_retrieval.download_utils import download_to_string
from src.data_retrieval.rate_limiter import RateLimiter

# See https://www.sitemaps.org/schemas/sitemap/0.9 — ElementTree
# requires this spelled out on every tag lookup, it isn't inferred from
# the document's own xmlns declaration. Same namespace/rationale as
# PitchstackDeckDownloader's _SITEMAP_XML_NAMESPACE.
_SITEMAP_XML_NAMESPACE = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# fabtcg.com's WAF 403s the default python-requests/curl User-Agent —
# see this module's docstring. Any ordinary browser-style string works;
# this one isn't otherwise significant.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_REQUEST_HEADERS = {"User-Agent": _BROWSER_USER_AGENT}

# Matches this source's numbered decklist sub-sitemaps
# (decklist-sitemap.xml, decklist-sitemap2.xml, ..., decklist-sitemap5.xml
# as of 2026-09-06) among the many unrelated sitemaps sitemap_index.xml
# also lists.
_DECKLIST_SITEMAP_PATTERN = re.compile(r"decklist-sitemap\d*\.xml$")

# Matches a real deck page URL exactly
# ("https://fabtcg.com/decklists/<slug>/") and nothing else — CONFIRMED
# LIVE that the decklist sitemaps also list a bare "/decklists/" (no
# slug) and locale-prefixed variants ("/fr/decklists/...",
# "/ja/decklists/..."), both deliberately excluded by anchoring to the
# bare-domain path.
_DECK_URL_PATTERN = re.compile(r"^https://fabtcg\.com/decklists/([^/]+)/$")

_DECKLIST_FRAGMENT_SELECTOR = "section.decklist-list-view.block.hidden"


class FabtcgDecklistDownloader:
    """Downloads Flesh and Blood decklist pages from fabtcg.com.

    Single-consumer to src/data_retrieval/fabtcg_decklists/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/fabtcg_decklists")
    DEFAULT_SITEMAP_INDEX_URL: ClassVar[str] = "https://fabtcg.com/sitemap_index.xml"

    def __init__(
        self,
        rate_limiter: RateLimiter,
        output_dir: Path | None = None,
        sitemap_index_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: paces every outgoing request this class
                makes. Passed in rather than constructed internally
                (dependency injection — PATTERNS.md), same convention
                as the sibling downloaders — so the same shared
                RateLimiter instance can be reused across sources, and
                so tests can supply a fast/no-op limiter.
            output_dir: directory this class's output is written into
                (deck_urls.txt, decklists/<slug>.html). Defaults to
                DEFAULT_RAW_DATA_DIR when omitted (expected to be a
                path under data/raw, per src/README.md — not this
                class's concern to enforce, just to receive).
            sitemap_index_url: URL of fabtcg.com's top-level sitemap
                index. Defaults to DEFAULT_SITEMAP_INDEX_URL when
                omitted.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/phase_2()
            are called.
        Exceptions: none.
        """
        self.rate_limiter = rate_limiter
        self.output_dir = output_dir or self.DEFAULT_RAW_DATA_DIR
        self.sitemap_index_url = sitemap_index_url or self.DEFAULT_SITEMAP_INDEX_URL

    def phase_1(self) -> Path:
        """Walk the sitemap index → decklist sub-sitemaps → deck page
        URLs, and write every deck URL found to disk.

        Inputs: none (uses self.sitemap_index_url, self.output_dir,
            self.rate_limiter).
        Output: path to the written URLs file
            (output_dir/deck_urls.txt).
        Side effects: one paced network request for the sitemap index,
            plus one per decklist sub-sitemap it lists (all via
            download_to_string(), which retries internally); creates
            output_dir if missing; writes one file
            (output_dir/deck_urls.txt), overwriting any existing one.
        Exceptions: raises on a request that still fails after
            download_to_string()'s retries, if a response body isn't
            valid XML, or if no deck URLs are found across every
            decklist sub-sitemap (a genuine site-format change should
            be loud, not silently produce an empty file).

        Example:
            >>> downloader = FabtcgDecklistDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> urls_path = downloader.phase_1()
        """
        result: Path
        deck_urls: list[str] = []

        # Fetch the sitemap index and find every decklist sub-sitemap
        # it lists (ignoring the many unrelated sitemaps also present).
        self.rate_limiter.wait()
        index_xml = download_to_string(self.sitemap_index_url, headers=_REQUEST_HEADERS)
        sub_sitemap_urls = self._extract_matching_loc_urls(
            index_xml, _DECKLIST_SITEMAP_PATTERN
        )

        # Fetch each decklist sub-sitemap and collect its deck URLs.
        for sub_sitemap_url in sub_sitemap_urls:
            self.rate_limiter.wait()
            sub_sitemap_xml = download_to_string(
                sub_sitemap_url, headers=_REQUEST_HEADERS
            )
            deck_urls.extend(
                self._extract_matching_loc_urls(sub_sitemap_xml, _DECK_URL_PATTERN)
            )

        if not deck_urls:
            raise ValueError(
                f"No deck URLs found across the decklist sitemaps listed in "
                f"{self.sitemap_index_url!r}"
            )

        # Dedup preserving first-seen order, same convention as the
        # sibling downloaders' phase_1 methods.
        deduplicated_urls = list(dict.fromkeys(deck_urls))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        result = self.output_dir / "deck_urls.txt"
        result.write_text(
            "".join(f"{deck_url}\n" for deck_url in deduplicated_urls),
            encoding="utf-8",
        )

        return result

    def phase_2(self) -> Path:
        """For every deck URL not already saved to disk, fetch that
        deck's page and save its extracted+prettified decklist
        fragment to its own file.

        Resumability check is "does decklists/<slug>.html already
        exist," not a shared manifest file — deliberately, since this
        phase's output is one file per deck rather than rows appended
        to one shared JSONL file (contrast
        PlayGwentDownloader.phase_2/PitchstackDeckDownloader.phase_2,
        which share one guides.jsonl/decks.jsonl and so need
        read_manifest()/append_with_manifest() to track which rows are
        already in it — there's no equivalent "which rows" question
        here, a file's mere existence already answers it). Same
        one-file-per-item resumability convention as
        SpireCodexRunDownloader.

        A deck page that still fails after download_to_string()'s
        retries, or whose decklist fragment can't be found, is logged
        via tqdm.write() and skipped, not raised — best-effort, same
        convention as every other two-phase downloader in this
        container: a run over many thousands of decks shouldn't be
        lost to one bad page.

        Inputs: none (uses self.output_dir, self.rate_limiter — reads
            output_dir/deck_urls.txt, written by phase_1()).
        Output: path to the directory decklist files are saved into
            (output_dir/decklists/).
        Side effects: one paced network request (more on retry — see
            download_to_string()) per deck URL not already saved;
            creates output_dir/decklists/ if missing; writes one file
            per successfully-fetched deck; prints a tqdm progress bar
            to stderr covering all deck URLs; prints one tqdm.write()
            line per deck that fails and gets skipped.
        Exceptions: raises only if output_dir/deck_urls.txt doesn't
            exist (phase_1() hasn't been run) — per-deck failures are
            caught internally (see above), never propagated.

        Example:
            >>> downloader = FabtcgDecklistDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> downloader.phase_1()
            >>> decklists_dir = downloader.phase_2()
        """
        result: Path

        urls_path = self.output_dir / "deck_urls.txt"
        deck_urls = [
            line
            for line in urls_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        result = self.output_dir / "decklists"
        result.mkdir(parents=True, exist_ok=True)

        # Fetch and save each deck page not already saved from a prior
        # run.
        for deck_url in tqdm(deck_urls, desc="fabtcg.com decklists", unit="deck"):
            slug = self._extract_deck_slug(deck_url)
            destination_path = result / f"{slug}.html"
            if destination_path.exists():
                continue

            self.rate_limiter.wait()
            try:
                page_html = download_to_string(deck_url, headers=_REQUEST_HEADERS)
                fragment_html = self._extract_decklist_fragment(page_html)
            except Exception as error:
                tqdm.write(f"fabtcg.com decklist {deck_url} failed: {error}")
                continue

            destination_path.write_text(fragment_html, encoding="utf-8")

        return result

    def _extract_matching_loc_urls(
        self, sitemap_xml: str, pattern: re.Pattern[str]
    ) -> list[str]:
        """Parse a sitemap XML document (index or leaf — both use the
        same <loc>-per-entry shape) and return every <loc> URL matching
        pattern, in document order.

        Private helper — used by phase_1() twice: once against the
        top-level sitemap index with _DECKLIST_SITEMAP_PATTERN (to find
        the decklist sub-sitemaps) and once per decklist sub-sitemap
        with _DECK_URL_PATTERN (to find deck page URLs). Both calls are
        the exact same parse-then-filter shape, differing only in which
        pattern is applied — centralized here rather than duplicated as
        two near-identical private methods (PRINCIPLES.md section 2).
        Split out from phase_1() itself so this parsing/filtering logic
        is independently testable against saved example XML without
        mocking network calls (same rationale as
        PitchstackDeckDownloader._extract_deck_ids).

        Namespace handling: sitemap XML uses the
        http://www.sitemaps.org/schemas/sitemap/0.9 namespace on every
        element, including <loc> — ElementTree requires this spelled
        out on every tag lookup, it isn't inferred from the document's
        own xmlns declaration (see PitchstackDeckDownloader's
        _SITEMAP_XML_NAMESPACE for the equivalent lookup dict this
        method's implementation will need).

        Inputs:
            sitemap_xml: raw XML body of a sitemap document (either the
                top-level index or one leaf sitemap — both list <loc>
                entries the same way).
            pattern: compiled regex a <loc> URL's text must fullmatch
                (or otherwise match, per the concrete implementation)
                to be included.
        Output: matching URLs, in the order their <loc> entries appear
            in the document. A <loc> URL not matching pattern is simply
            skipped, not an error — every sitemap this is called against
            is expected to also list entries the caller doesn't want.
        Side effects: none — pure parsing, no network or disk I/O.
        Exceptions: raises xml.etree.ElementTree.ParseError if
            sitemap_xml isn't valid XML.
        """
        root = ElementTree.fromstring(sitemap_xml)
        loc_elements = root.findall(".//sitemap:loc", _SITEMAP_XML_NAMESPACE)

        matching_urls = []
        for loc_element in loc_elements:
            if loc_element.text is not None and pattern.search(loc_element.text):
                matching_urls.append(loc_element.text)

        return matching_urls

    def _extract_deck_slug(self, deck_url: str) -> str:
        """Extract a deck page URL's slug (the path segment identifying
        the deck, used as its saved filename).

        Private helper — single consumer is phase_2(). Split out purely
        for readability at the phase_2() call site, not for
        independent testability the way the two sitemap-parsing helpers
        above are (this one has no interesting failure mode worth its
        own test file).

        Inputs:
            deck_url: a deck page URL matching _DECK_URL_PATTERN's
                shape (i.e. one phase_1() already wrote to
                deck_urls.txt).
        Output: the slug, e.g. "tom-penny-go-wide-warrior-deck" for
            "https://fabtcg.com/decklists/tom-penny-go-wide-warrior-deck/".
        Side effects: none.
        Exceptions: raises ValueError if deck_url doesn't match
            _DECK_URL_PATTERN's shape — every URL reaching this method
            is expected to already be one phase_1() wrote, so a
            mismatch here is a genuine bug, not a data quirk to
            tolerate.
        """
        match = _DECK_URL_PATTERN.match(deck_url)
        if match is None:
            raise ValueError(
                f"{deck_url!r} does not match the expected deck page URL shape"
            )

        return match.group(1)

    def _extract_decklist_fragment(self, page_html: str) -> str:
        """Extract and pretty-print a deck page's decklist fragment
        ("<section class="decklist-list-view block hidden">...").

        Private helper — single consumer is phase_2(). Split out so
        the extraction logic is independently testable against saved
        example HTML (data/tmp/list_view.html,
        data/tmp/Tom Penny Go Wide Warrior Deck - Flesh and Blood
        TCG.html) without mocking network calls (same rationale as
        PlayGwentDownloader._extract_guide_payload).

        Parsed with BeautifulSoup (html.parser backend, matching this
        project's existing convention — see
        play_gwent/downloader.py and
        data_refinement/card_binder/gwent_one/ingestion_stage.py) and
        located via _DECKLIST_FRAGMENT_SELECTOR. Pretty-printed via
        BeautifulSoup's own .prettify() — this container's only
        transformation of the fragment's contents (see this module's
        docstring), not a reshape.

        Inputs:
            page_html: raw HTML of a deck detail page.
        Output: the fragment's prettified HTML, as a string.
        Side effects: none — pure parsing, no network or disk I/O.
        Exceptions: raises ValueError if no element matches
            _DECKLIST_FRAGMENT_SELECTOR — a genuine site-format change
            should be loud, not silently produce an empty/wrong file.
        """
        soup = BeautifulSoup(page_html, "html.parser")
        fragment = soup.select_one(_DECKLIST_FRAGMENT_SELECTOR)
        if fragment is None:
            raise ValueError(
                f"No element matching {_DECKLIST_FRAGMENT_SELECTOR!r} found in "
                "decklist page"
            )

        return fragment.prettify()
