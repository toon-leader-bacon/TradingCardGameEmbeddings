"""Downloads MTGTop8 tournament decks (MTG), one format at a time: the
format's search listing, then each listed deck's MTGO text export.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Unwrapping is reading
the search table into ListingRows (listing.py) and checking the export
text (deck_export.py); reading the export into cards, and the
sideboard-or-command-zone question, are data_refinement's. The crawl's
files, walk loop and HTTP reading are the shared resumable_crawl/
package's.

TERMS AND PACE (read 2026-11-11): robots.txt disallows only /compare
and sets "Crawl-delay: 5" for "User-agent: *" (Googlebot's entry has no
delay), so the default pace is DEFAULT_REQUESTS_PER_MINUTE (12/min). A
caller may pass a faster RateLimiter (--rpm), e.g. after asking the
site for more; the downloader logs a warning when it does, and does
not refuse. No AI or scraping clause, no terms page; the footer is the
WotC fan-content notice. Player names are not kept.

ACCESS (probed 2026-11-11): no sitemap (404) and no bulk export.
/search?format=<code>&current_page=<n> (GET answers as the site's POST
form does) lists 25 decks a page, newest first, with deck id, event id,
deck name, event name, rank and date; a page past the end lists none.
/mtgo?d=<id> is the deck as MTGO text (~0.5 KB), with the sideboard
after a "Sideboard" line; in the Commander formats that section holds
the commander(s). ~634k decks in all 20 formats: at 12/min, ~37 days
of exports plus ~35 h of listing pages.

LAYOUT: one directory per format, raw_data_dir/<format code>/, each a
whole resumable crawl (listing/, crawl_plan.jsonl, cursor.json,
rejects.jsonl, decks-<nnnnn>.jsonl(.gz)). A deck lands as its
ListingRow's fields plus "mtgo" (the export text).

    1. phase_1() lands the format's search pages in order as
       listing/page-<nnnnn>.jsonl (one paced GET each, resumable),
       until a page lists fewer than SEARCH_PAGE_ROWS decks, then merges
       them into crawl_plan.jsonl (crawl_plan.py). Forward paging means
       decks added mid-listing repeat a deck rather than skip one. An
       existing plan is kept: decks listed after it are a later plan's.
       A cursor without a plan is refused.
    2. phase_2() walks the plan from cursor.json (ResumableCrawl), one
       paced GET per deck, for up to max_decks_per_session decks: an
       export lands as one row; an empty answer lands nothing; a 4xx
       (but 429) is a reject; network errors, 5xx, a 429 past its waits
       and a body that isn't an export are transient.
       _MAX_CONSECUTIVE_REJECTS rejects in a row stop the session.
"""

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar

from tqdm import tqdm

from src.data_retrieval.downloader import Downloader
from src.data_retrieval.mtgtop8.crawl_plan import (
    PLAN_NAME,
    MtgTop8PlanFile,
    merge_and_shuffle,
    read_listing_rows,
    write_listing_rows,
)
from src.data_retrieval.mtgtop8.deck_export import read_deck_export
from src.data_retrieval.mtgtop8.formats import MtgTop8Format
from src.data_retrieval.mtgtop8.listing import (
    SEARCH_PAGE_ROWS,
    ListingRow,
    parse_search_page,
)
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.resumable_crawl import (
    PERMANENT_STATUSES,
    FetchOutcome,
    ItemFound,
    ItemNotFound,
    ResumableCrawl,
    WalkPolicy,
    crawl_get,
    read_outcome,
    refuse_orphan_cursor,
)

_logger = logging.getLogger(__name__)

_SITE_URL = "https://www.mtgtop8.com"
_LISTING_DIR_NAME = "listing"
# A landed page's file name (_landed_page_path writes it)
_LANDED_PAGE_PATTERN = re.compile(r"^page-(\d{5})\.jsonl$")
_MAX_CONSECUTIVE_REJECTS = 20
# The default pace: robots.txt's "Crawl-delay: 5" (2026-11-11). Only a
# default: a caller's RateLimiter may be faster (with a warning)
DEFAULT_REQUESTS_PER_MINUTE = 12.0


class MtgTop8DeckDownloader(Downloader):
    """Lists one MTGTop8 format's decks, then lands each deck's export
    with its listing row.

    Single-consumer to src/data_refinement/deck_box/mtgtop8_decks/.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/mtgtop8")

    def __init__(
        self,
        deck_format: MtgTop8Format,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        site_url: str | None = None,
        max_decks_per_session: int | None = None,
    ) -> None:
        """
        Inputs:
            deck_format: the format to crawl; its files go in
                raw_data_dir/<code>/.
            rate_limiter: see Downloader.__init__; defaults to
                DEFAULT_REQUESTS_PER_MINUTE, not 60. Any rate is taken;
                one faster than the default is logged as a warning. Pass
                one limiter to every format of a session so they share
                the pace.
            raw_data_dir: the root holding every format's directory.
            site_url: the site root, no trailing slash.
            max_decks_per_session: decks one phase_2() call fetches; None
                walks to the end of the plan.
        Output: none.
        Side effects: a warning log if rate_limiter is faster than
            DEFAULT_REQUESTS_PER_MINUTE; no I/O until phase_1().
        Exceptions: ValueError if max_decks_per_session < 1.
        """
        super().__init__(
            rate_limiter or RateLimiter(DEFAULT_REQUESTS_PER_MINUTE), raw_data_dir
        )
        if self.rate_limiter.requests_per_minute > DEFAULT_REQUESTS_PER_MINUTE:
            _logger.warning(
                "MTGTop8 at %s/min, faster than robots.txt's Crawl-delay (%s/min)",
                self.rate_limiter.requests_per_minute,
                DEFAULT_REQUESTS_PER_MINUTE,
            )
        if max_decks_per_session is not None and max_decks_per_session < 1:
            raise ValueError(
                f"max_decks_per_session must be >= 1, got {max_decks_per_session}"
            )
        self.deck_format = deck_format
        self.site_url = site_url or _SITE_URL
        self.max_decks_per_session = max_decks_per_session
        self.format_dir = self.raw_data_dir / deck_format.value

    def _run_phase_1(self) -> Path:
        """Land the format's search pages not yet on disk, then write
        crawl_plan.jsonl, unless the plan exists.

        Inputs: none. Output: the plan's path.
        Side effects: one paced GET per page not yet landed (a tqdm bar
            counting pages; the listing's length is unknown until its
            short page); writes listing/*.jsonl and the plan.
        Exceptions: requests.RequestException after retries; ValueError
            if a page isn't a results page, or the format lists no
            decks; RuntimeError if a cursor exists without a plan.

        Example:
            >>> MtgTop8DeckDownloader(MtgTop8Format.PAUPER).phase_1()
            PosixPath('data/raw/mtgtop8/PAU/crawl_plan.jsonl')
        """
        plan_path = self.format_dir / PLAN_NAME
        if plan_path.is_file():
            return plan_path
        refuse_orphan_cursor(self.format_dir, PLAN_NAME)
        listing_dir = self.format_dir / _LISTING_DIR_NAME
        listing_dir.mkdir(parents=True, exist_ok=True)

        # Land pages from the first one not on disk, until a short page
        page_number = _next_page_to_land(listing_dir)
        with tqdm(
            desc=f"MTGTop8 {self.deck_format.value} listing",
            unit="page",
            initial=len(_landed_page_paths(listing_dir)),
        ) as progress:
            while page_number is not None:
                rows = self._land_search_page(page_number, listing_dir)
                progress.update(1)
                full_page = len(rows) == SEARCH_PAGE_ROWS
                page_number = page_number + 1 if full_page else None

        # Merge into the plan: unique decks, shuffled
        landed_paths = _landed_page_paths(listing_dir)
        rows = merge_and_shuffle(landed_paths)
        _logger.info(
            "MTGTop8 %s plan: %d decks from %d pages",
            self.deck_format.value,
            len(rows),
            len(landed_paths),
        )
        write_listing_rows(plan_path, rows)
        return plan_path

    def phase_2(self) -> Path:
        """Walk the format's plan from the cursor, for up to
        max_decks_per_session decks (see ResumableCrawl.walk).

        Inputs: none (reads crawl_plan.jsonl and cursor.json).
        Output: the format's directory.
        Side effects: one paced GET per deck (more on a retry); shard and
            reject rows; cursor.json after each deck; a tqdm bar.
        Exceptions: FileNotFoundError if phase_1() never wrote a plan;
            ValueError if the cursor belongs to another plan, or a 200
            body isn't an export; requests.RequestException for a
            transient failure past its retries; CrawlBlockedError after
            _MAX_CONSECUTIVE_REJECTS rejects in a row.

        Example:
            >>> MtgTop8DeckDownloader(MtgTop8Format.PAUPER,
            ...     max_decks_per_session=720).phase_2()
            PosixPath('data/raw/mtgtop8/PAU')
        """
        plan = MtgTop8PlanFile(self.format_dir / PLAN_NAME)
        crawl = ResumableCrawl[int](
            self.format_dir,
            plan,
            f"MTGTop8 {self.deck_format.value} decks",
            WalkPolicy(max_consecutive_rejects=_MAX_CONSECUTIVE_REJECTS),
        )
        crawl.walk(
            lambda deck_id: self._fetch_deck(plan.row_for(deck_id)),
            self.max_decks_per_session,
        )
        return self.format_dir

    def _land_search_page(
        self, page_number: int, listing_dir: Path
    ) -> list[ListingRow]:
        """GET one search page of this format and write its rows to
        listing/page-<nnnnn>.jsonl, atomically, so a landed page is
        always whole (an empty last page lands as an empty file).
        Inputs: page_number (>= 1), listing_dir.
        Output: the page's rows.
        Side effects: one paced GET (crawl_get); writes the page file.
        Exceptions: requests.RequestException (an error status, or
            transient); ValueError if it isn't a results page; OSError."""
        response = crawl_get(
            f"{self.site_url}/search?format={self.deck_format.value}"
            f"&current_page={page_number}",
            self.rate_limiter,
        )
        result = parse_search_page(response.text)
        write_listing_rows(_landed_page_path(listing_dir, page_number), result)
        return result

    def _fetch_deck(self, listing: ListingRow) -> FetchOutcome:
        """GET /mtgo?d=<deck id> (crawl_get, tolerating
        PERMANENT_STATUSES), read by read_outcome with read_deck_export:
        an export -> ItemFound(listing's fields + "mtgo"); an empty body
        -> ItemNotFound.
        Inputs: listing (the deck's plan row). Output: FetchOutcome.
        Side effects: one paced GET (more on a retry).
        Exceptions: requests.RequestException (transient); ValueError for
            a 200 that isn't an export (transient)."""
        response = crawl_get(
            f"{self.site_url}/mtgo?d={listing.deck_id}",
            self.rate_limiter,
            PERMANENT_STATUSES,
        )
        return read_outcome(
            response,
            lambda export: _deck_outcome(listing, export),
            f"deck {listing.deck_id}",
            read_deck_export,
        )


def _deck_outcome(listing: ListingRow, export: str | None) -> FetchOutcome:
    """A read export as a FetchOutcome: None (the empty answer for a
    missing deck) -> ItemNotFound; text -> ItemFound(listing's fields
    plus "mtgo": the text).
    Inputs: listing, export. Output: FetchOutcome. Side effects: none.
    Exceptions: none."""
    if export is None:
        return ItemNotFound()
    return ItemFound({**asdict(listing), "mtgo": export})


def _next_page_to_land(listing_dir: Path) -> int | None:
    """The page phase 1 fetches next: 1 when none is landed; None when
    the highest landed page is short (the listing is complete); else
    the highest landed page + 1.
    Inputs: listing_dir. Output: page number or None.
    Side effects: reads the highest landed page.
    Exceptions: OSError; ValueError for a malformed page file."""
    landed_paths = _landed_page_paths(listing_dir)
    if not landed_paths:
        return 1
    last_path = landed_paths[-1]
    if len(read_listing_rows(last_path)) < SEARCH_PAGE_ROWS:
        return None
    return _page_number_of(last_path) + 1


def _landed_page_path(listing_dir: Path, page_number: int) -> Path:
    """Where search page page_number lands. Inputs: listing_dir,
    page_number. Output: <listing_dir>/page-<nnnnn>.jsonl.
    Side effects: none. Exceptions: none."""
    return listing_dir / f"page-{page_number:05d}.jsonl"


def _landed_page_paths(listing_dir: Path) -> list[Path]:
    """Every landed page (_LANDED_PAGE_PATTERN), in page-number order.
    Inputs: listing_dir. Output: list of paths. Side effects: lists it.
    Exceptions: OSError."""
    landed_paths = [
        path for path in listing_dir.iterdir() if _LANDED_PAGE_PATTERN.match(path.name)
    ]
    return sorted(landed_paths, key=_page_number_of)


def _page_number_of(landed_path: Path) -> int:
    """The page number a _landed_page_path() path names.
    Inputs: landed_path. Output: int >= 1. Side effects: none.
    Exceptions: ValueError if the name isn't page-<nnnnn>.jsonl."""
    match = _LANDED_PAGE_PATTERN.match(landed_path.name)
    if match is None:
        raise ValueError(f"not a landed page: {landed_path.name!r}")
    return int(match.group(1))
