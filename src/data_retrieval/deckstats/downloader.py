"""Downloads Deckstats decks (MTG) one deck page at a time: the decks
Deckstats' sitemaps list, then every deck id in a fixed spread order.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Deckstats is a
Laravel + Inertia.js site: each page embeds its full state as JSON in
<script data-page="app">, so unwrapping is reading that JSON (page.py)
and pruning it (pruning.py), with no HTML scraping. Refining (formats,
zones, card matching) is data_refinement's job. The crawl's files,
walk loop and HTTP reading are the shared resumable_crawl/ package's.

TERMS AND PACE (read 2026-11-06): robots.txt has no AI clause; it
disallows /api/, /search, playtest and proxy pages, and every URL with
a query string (/*?), and allows deck pages and the sitemaps. The
Terms' "Public content and data" section licenses public decks "for
research and analysis purposes, including by providing that data to
third parties for their own research and analysis". The site is behind
Cloudflare; a plain User-Agent got 200s with no challenge. Pace is the
project's 60/min. Nothing about deck owners (names, ids, avatars,
folders) is kept.

ACCESS (probed 2026-11-06): no bulk route (the API is disallowed; the
text export needs a query string). A deck page is ~79 KB of HTML
(~19 KB gzipped on the wire); a pruned row is ~6.5 KB for a 60-card
deck. Deck URLs need the name slug: /decks/<id> and /decks/<id>-<any>
answer 301 to /decks/<id>-<slug>. A private deck answers 200 with
component "DeckPrivate" (its name only). The sitemap index lists 7
deck sitemaps (sitemap-decks-qualified-00..04, -evergreen-00,
-fresh-00): ~225k decks, ids 325 to ~4.52M.

PRIVATE DECKS (the user's choice): a private deck's page carries only
its name; it lands as {"idsdeck", "deckName", "visibility": "private"}
in private/decks-<nnnnn>.jsonl(.gz), apart from the readable decks, so
a later pass knows the id exists without re-probing it. Every readable
deck keeps its "visibility" as a flag.

    1. phase_1() fetches the sitemap index and its deck sitemaps (not
       the Lorcana one) and writes crawl_plan.json
       (crawl_plan.py, a SitemapThenSpreadPlan): the sitemap's deck
       paths ("<id>-<slug>") first, then every id in [1, the highest
       sitemap id] in a fixed spread order (bare "<id>"), skipping
       sitemap ids. An existing plan is kept; a cursor without a plan
       is refused.
    2. phase_2() walks the plan from cursor.json (ResumableCrawl), for
       up to max_decks_per_session items. Every item is fetched the
       same way (_follow_deck_redirect): one paced GET of
       /decks/<item> that does not follow redirects; a 3xx to a
       Deckstats /decks/ URL gets exactly one more paced GET, also not
       following redirects. A sitemap path with a current slug costs one
       request; a bare id, or a renamed deck, costs two, both paced.
       A Deck page lands as one pruned row in decks-<nnnnn>.jsonl(.gz);
       a DeckPrivate page as one row in private/; a 404 lands nothing;
       another 4xx (but 429), a redirect not taken (off /decks/, a
       second hop, no Location, a query string, fragment or dot
       segment), another page component, or an
       unreadable page is a reject; network errors, 5xx, a 429 beyond
       its waits and a 200 without the page JSON (a challenge page) are
       transient. _MAX_CONSECUTIVE_REJECTS rejects in a row stop the
       session.
"""

import logging
from pathlib import Path
from typing import ClassVar
from urllib.parse import urljoin, urlsplit

import requests

from src.data_retrieval.deckstats.crawl_plan import (
    DECKSTATS_ITEMS,
    PLAN_NAME,
    keep_first_path_per_id,
    parse_deck_sitemap_urls,
    parse_sitemap_deck_paths,
)
from src.data_retrieval.deckstats.page import parse_inertia_page
from src.data_retrieval.deckstats.pruning import classify_page
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.resumable_crawl import (
    PERMANENT_STATUSES,
    REDIRECT_STATUSES,
    FetchOutcome,
    ResumableCrawl,
    SitemapThenSpreadPlan,
    WalkPolicy,
    crawl_get,
    read_outcome,
    refuse_orphan_cursor,
    write_text_atomically,
)

_logger = logging.getLogger(__name__)

_SITE_URL = "https://deckstats.net"
_MAX_CONSECUTIVE_REJECTS = 20


class DeckstatsDeckDownloader(Downloader):
    """Walks Deckstats' sitemap decks, then every deck id, and lands one
    pruned row per readable deck (one name-only row per private deck).

    Single-consumer to src/data_refinement/deck_box/deckstats/ (not yet
    built).
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/deckstats")

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        site_url: str | None = None,
        max_decks_per_session: int | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter, raw_data_dir: see Downloader.__init__ (60/min).
            site_url: the site root, no trailing slash.
            max_decks_per_session: plan items one phase_2() call fetches
                (a redirect to a deck URL adds a second GET); None walks to
                the end of the plan.
        Output: none. Side effects: none; no I/O until phase_1().
        Exceptions: ValueError if max_decks_per_session < 1.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.site_url = site_url or _SITE_URL
        if max_decks_per_session is not None and max_decks_per_session < 1:
            raise ValueError(
                f"max_decks_per_session must be >= 1, got {max_decks_per_session}"
            )
        self.max_decks_per_session = max_decks_per_session

    def _run_phase_1(self) -> Path:
        """Write crawl_plan.json unless it exists.

        Inputs: none. Output: the plan's path.
        Side effects: one paced GET for sitemap.xml and one per deck
            sitemap (7 on 2026-11-06); writes the plan.
        Exceptions: requests.RequestException after retries; ValueError
            if the index lists no deck sitemaps or they list no decks;
            RuntimeError if a cursor exists without a plan.

        Example:
            >>> DeckstatsDeckDownloader().phase_1()
            PosixPath('data/raw/deckstats/crawl_plan.json')
        """
        plan_path = self.raw_data_dir / PLAN_NAME
        if plan_path.is_file():
            return plan_path
        refuse_orphan_cursor(self.raw_data_dir, PLAN_NAME)

        # Gather the sitemap's deck paths, each id's first listing
        index_xml = crawl_get(f"{self.site_url}/sitemap.xml", self.rate_limiter).text
        listed_paths: list[str] = []
        for url in parse_deck_sitemap_urls(index_xml):
            listed_paths.extend(
                parse_sitemap_deck_paths(crawl_get(url, self.rate_limiter).text)
            )
        sitemap_paths = keep_first_path_per_id(listed_paths)

        # Then every id up to the highest the sitemaps list
        plan = SitemapThenSpreadPlan.over_sitemap(sitemap_paths, DECKSTATS_ITEMS)
        _logger.info(
            "Deckstats plan: %d sitemap decks, then ids 1..%d (stride %d)",
            len(sitemap_paths),
            plan.spread.n,
            plan.spread.stride,
        )
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomically(plan_path, plan.to_json())
        return plan_path

    def phase_2(self) -> Path:
        """Walk the plan from the cursor, for up to max_decks_per_session
        items (see resumable_crawl.ResumableCrawl.walk).

        Inputs: none (reads crawl_plan.json and cursor.json).
        Output: raw_data_dir.
        Side effects: one or two paced GETs per item (more on a retry);
            shard rows (private/ for private decks) and reject rows;
            cursor.json after each item; a tqdm bar.
        Exceptions: FileNotFoundError if phase_1() never wrote a plan;
            ValueError if the cursor belongs to another plan, or a 200
            page has no page JSON; requests.RequestException for a
            transient failure past its retries; CrawlBlockedError after
            _MAX_CONSECUTIVE_REJECTS rejects in a row.

        Example:
            >>> DeckstatsDeckDownloader(max_decks_per_session=3600).phase_2()
            PosixPath('data/raw/deckstats')
        """
        plan = SitemapThenSpreadPlan.from_json(
            (self.raw_data_dir / PLAN_NAME).read_text(encoding="utf-8"), DECKSTATS_ITEMS
        )
        crawl = ResumableCrawl[str](
            self.raw_data_dir,
            plan,
            "Deckstats decks",
            WalkPolicy(max_consecutive_rejects=_MAX_CONSECUTIVE_REJECTS),
        )
        crawl.walk(self._fetch_deck, self.max_decks_per_session)
        return self.raw_data_dir

    def _fetch_deck(self, deck_path: str) -> FetchOutcome:
        """Fetch one plan item's deck page (_follow_deck_redirect) and read
        it (read_outcome with parse_inertia_page and classify_page).

        Inputs: deck_path ("<id>-<slug>" or a bare "<id>").
        Output: FetchOutcome.
        Side effects: one or two paced GETs (more on a retry).
        Exceptions: requests.RequestException (transient); ValueError for
            a 200 without the page JSON (transient)."""
        response = self._follow_deck_redirect(f"{self.site_url}/decks/{deck_path}")
        return read_outcome(
            response, classify_page, f"deck {deck_path}", parse_inertia_page
        )

    def _follow_deck_redirect(self, url: str) -> requests.Response:
        """GET url (tolerating PERMANENT_STATUSES, not following
        redirects). If it answers a REDIRECT_STATUSES status whose
        Location makes a deck URL on this site (_deck_redirect_target),
        GET exactly that URL once more, the same way; otherwise return
        the first answer. At most one hop: a redirect not taken (off
        /decks/, a query string, no Location, or a second hop) comes back
        as a 3xx, which read_outcome rejects.

        Inputs: url (site_url/decks/<item>).
        Output: the last response (a 200 page, a 4xx, or an untaken 3xx).
        Side effects: one or two paced GETs (more on a retry).
        Exceptions: requests.RequestException (transient)."""
        response = self._get_without_redirects(url)
        if response.status_code not in REDIRECT_STATUSES:
            return response
        target = self._deck_redirect_target(url, response.headers.get("Location", ""))
        if target is None:
            return response
        return self._get_without_redirects(target)

    def _get_without_redirects(self, url: str) -> requests.Response:
        """One paced GET that tolerates PERMANENT_STATUSES and doesn't
        follow redirects.
        Inputs: url. Output: the response (2xx, 3xx or a tolerated 4xx).
        Side effects: one paced GET (more on a retry).
        Exceptions: requests.RequestException (transient)."""
        return crawl_get(
            url, self.rate_limiter, PERMANENT_STATUSES, follow_redirects=False
        )

    def _deck_redirect_target(self, url: str, location: str) -> str | None:
        """The URL a redirect from url to location leads to, if it is a
        deck page this crawler may fetch: this site (same scheme and
        host), a path under /decks/ with no "." or ".." segment, and no
        query string or fragment (robots.txt disallows every query
        string). The URL checked is the URL fetched.

        Inputs: url (the redirecting request), location (its Location
            header, absolute, root-relative or relative; may be empty).
        Output: the target URL, or None. Side effects: none.
        Exceptions: none.

        Example:
            >>> DeckstatsDeckDownloader()._deck_redirect_target(
            ...     "https://deckstats.net/decks/325", "/decks/325-Burn")
            'https://deckstats.net/decks/325-Burn'
        """
        if not location:
            return None
        target = urljoin(url, location)
        parts = urlsplit(target)
        site = urlsplit(self.site_url)
        segments = parts.path.split("/")
        if (
            (parts.scheme, parts.netloc) != (site.scheme, site.netloc)
            or not parts.path.startswith("/decks/")
            or "." in segments
            or ".." in segments
            or parts.query
            or parts.fragment
        ):
            return None
        return target
