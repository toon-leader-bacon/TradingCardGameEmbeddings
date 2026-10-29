"""Downloads Hearthstone deck pages from hearthstonetopdecks.com, in two
phases.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Deck pages are full
HTML, so phase 2 unwraps each page to the few pieces that hold its data
(title, deck code, the details block), as play_gwent unwraps its JSON.
The deck code is kept as the opaque string the site shows; decoding it
is data_refinement's job.

HearthstoneTopDecks hosts user-submitted decks plus staff-curated
tournament and streamer lists, 2014 to today. robots.txt allows
everything but /api/ (not used here). CONFIRMED LIVE 2026-10-28:

    1. phase_1() walks the deck listing (/decks/page/<n>/, 25 decks a
       page, newest first) until a page 404s or lists no decks,
       writing every deck page URL to deck_urls.txt: ~4,755 pages,
       ~118,900 decks, about 80 minutes.
    2. phase_2() fetches every deck page not yet in decks_manifest.txt
       and appends one row per page to decks.jsonl:
       {"url", "title", "deck_code", "details": {label: value}}.
       details is the page's header block verbatim as label/text pairs
       ("Posted by", "Published", "Updated", "Dust", "Rating", "Class",
       "Format", "Type", "Style", "Season"; old pages lack some).
       deck_code is null on a page without one, so that page isn't
       refetched forever.

Why the listing and not the sitemaps: /sitemap.xml lists 12 deck
sitemaps (/sitemaps/decks-<n>.xml) with the same ~118,900 URLs in 13
requests, but on 2026-10-28 decks-5 and decks-8 answered HTTP 500 on
every try (and once 200 with no URLs), losing ~20,000 decks. The
listing reaches every deck. A deck posted mid-walk shifts later pages
by one, so a deck can be missed at a page boundary; a re-run of
phase 1 picks it up.

Phase 2 is ~119k pages at the usual 60/min, about 33 hours.
"""

import json
import logging
import re
from pathlib import Path
from typing import ClassVar

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from src.data_retrieval.download_utils import (
    append_with_manifest,
    call_with_retries,
    read_manifest,
)
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 60
_MAX_FETCH_ATTEMPTS = 3  # 1 initial try + up to 2 retries
# A deck page link on a listing page: href="/decks/<slug>/" (the
# listing's own /decks/page/<n>/ links have a second path segment)
_DECK_LINK_PATTERN = re.compile(r'href="(/decks/[^"/]+/)"')
_TITLE_SELECTOR = "article h1"
_DECK_CODE_SELECTOR = 'article input[aria-label="Deck code"]'
# Each detail is a <span>Label: <strong>value</strong></span> in the
# header block's <p> rows
_DETAIL_SELECTOR = "article div.text-center > p > span"


class HearthstoneTopDecksDownloader(Downloader):
    """Downloads deck page URLs (the deck listing) and deck pages from
    hearthstonetopdecks.com.

    Single-consumer to src/data_retrieval/hearthstonetopdecks/.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/hearthstonetopdecks")
    DEFAULT_SITE_URL: ClassVar[str] = "https://www.hearthstonetopdecks.com"
    DECKS_FILENAME: ClassVar[str] = "decks.jsonl"

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        site_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
            site_url: the site root, no trailing slash. Defaults to
                DEFAULT_SITE_URL.
        Output: none (constructor).
        Side effects: none; no I/O until phase_1()/phase_2().
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.site_url = site_url or self.DEFAULT_SITE_URL

    @property
    def deck_urls_path(self) -> Path:
        """raw_data_dir/deck_urls.txt, phase_1()'s output."""
        return self.raw_data_dir / "deck_urls.txt"

    @property
    def decks_path(self) -> Path:
        """raw_data_dir/decks.jsonl, phase_2()'s output."""
        return self.raw_data_dir / self.DECKS_FILENAME

    @property
    def decks_manifest_path(self) -> Path:
        """raw_data_dir/decks_manifest.txt, the URLs phase_2() has done."""
        return self.raw_data_dir / "decks_manifest.txt"

    def _run_phase_1(self) -> Path:
        """Walk the deck listing and write every deck page URL to
        deck_urls.txt.

        Inputs: none (uses self.site_url).
        Output: self.deck_urls_path.
        Side effects: one paced request per listing page (more on
            retry); creates raw_data_dir; overwrites deck_urls.txt;
            shows a tqdm bar.
        Exceptions: requests.RequestException if a listing page still
            fails after retries (a page walk can't skip a page);
            ValueError if no decks were found at all.

        Example:
            >>> HearthstoneTopDecksDownloader().phase_1()
        """
        deck_urls: list[str] = []

        # Page until a 404 or a page with no deck links
        page = 1
        with tqdm(desc="HSTD deck listing", unit="page") as progress:
            while True:
                page_deck_urls = self._fetch_listing_page(page)
                if not page_deck_urls:
                    break
                deck_urls.extend(page_deck_urls)
                progress.update(1)
                page += 1

        if not deck_urls:
            raise ValueError(f"no decks listed at {self.site_url}/decks/")

        # Dedup, keeping walk order: a deck posted mid-walk repeats one
        self._write_deck_urls(list(dict.fromkeys(deck_urls)))
        return self.deck_urls_path

    def phase_2(self) -> Path:
        """Fetch every deck page not already in decks_manifest.txt and
        append its unwrapped row to decks.jsonl.

        Best-effort per page: one that still fails after retries is
        logged and skipped, so a later run retries it.

        Inputs: none (reads deck_urls.txt from phase_1()).
        Output: self.decks_path.
        Side effects: one paced request per new page; appends to
            decks.jsonl and decks_manifest.txt (data row first, see
            append_with_manifest); shows a tqdm bar; logs a warning per
            skipped page.
        Exceptions: FileNotFoundError if deck_urls.txt is missing.

        Example:
            >>> downloader = HearthstoneTopDecksDownloader()
            >>> downloader.phase_1()
            >>> downloader.phase_2()
        """
        deck_urls = [
            line
            for line in self.deck_urls_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        already_downloaded = read_manifest(self.decks_manifest_path)

        # Fetch and unwrap each new deck page
        for deck_url in tqdm(deck_urls, desc="HSTD deck pages", unit="deck"):
            if deck_url in already_downloaded:
                continue
            try:
                page_html = self._get_with_retries(deck_url).text
            except requests.RequestException as error:
                _logger.warning("HSTD deck page %s failed: %s", deck_url, error)
                continue
            row = unwrap_deck_page(deck_url, page_html)
            append_with_manifest(
                self.decks_path, self.decks_manifest_path, json.dumps(row), deck_url
            )

        return self.decks_path

    def _fetch_listing_page(self, page: int) -> list[str]:
        """One listing page's deck page URLs, in page order; [] past the
        last page (a 404).

        Inputs: page (1-based). Output: absolute deck page URLs.
        Side effects: paced request(s), see _get_with_retries.
        Exceptions: as _get_with_retries.
        """
        response = self._get_with_retries(
            f"{self.site_url}/decks/page/{page}/", allow_not_found=True
        )
        if response.status_code == 404:
            return []
        deck_paths = _DECK_LINK_PATTERN.findall(response.text)
        return [f"{self.site_url}{path}" for path in dict.fromkeys(deck_paths)]

    def _get_with_retries(
        self, url: str, allow_not_found: bool = False
    ) -> requests.Response:
        """GET url, paced, retrying via call_with_retries.

        Inputs: url; allow_not_found (return a 404 instead of raising,
            and don't retry it).
        Output: the 2xx response (or the 404, if allowed).
        Side effects: paced request(s); sleeps between attempts.
        Exceptions: the last requests.RequestException.
        """

        def _attempt() -> requests.Response:
            self.rate_limiter.wait()
            response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
            if not (allow_not_found and response.status_code == 404):
                response.raise_for_status()
            return response

        return call_with_retries(_attempt, max_attempts=_MAX_FETCH_ATTEMPTS)

    def _write_deck_urls(self, deck_urls: list[str]) -> None:
        """Overwrite deck_urls.txt with one URL per line.

        Inputs: deck_urls. Output: none.
        Side effects: creates raw_data_dir; writes deck_urls.txt.
        Exceptions: OSError on a write failure.
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.deck_urls_path.write_text(
            "".join(f"{url}\n" for url in deck_urls), encoding="utf-8"
        )


def unwrap_deck_page(url: str, page_html: str) -> dict:
    """The raw row for one deck page: its url, title, deck code and
    details block, all as the page shows them.

    Public (not a method) so tests and data_refinement fixtures can
    build rows from saved HTML without a downloader.

    Inputs: url (the page's URL), page_html.
    Output: {"url", "title", "deck_code", "details"}; title and
        deck_code are None when the page lacks them.
    Side effects: none. Exceptions: none.

    Example:
        >>> unwrap_deck_page(url, html)["deck_code"]
        'AAECAaIHCLICyAPn...'
    """
    soup = BeautifulSoup(page_html, "html.parser")
    title = soup.select_one(_TITLE_SELECTOR)
    deck_code = soup.select_one(_DECK_CODE_SELECTOR)
    return {
        "url": url,
        "title": title.get_text(strip=True) if title else None,
        "deck_code": _read_input_value(deck_code),
        "details": _read_details(soup),
    }


def _read_input_value(element: Tag | None) -> str | None:
    """An <input>'s non-empty value attribute, else None.

    Inputs: element (None when the page has no such input).
    Output: str | None. Side effects: none. Exceptions: none.
    """
    if element is None:
        return None
    value = element.get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _read_details(soup: BeautifulSoup) -> dict[str, str]:
    """The header block's "Label: value" spans as {label: value}.

    Inputs: soup (a parsed deck page). Output: {label: value}; {} when
        the page has no header block.
    Side effects: none. Exceptions: none.
    """
    result: dict[str, str] = {}
    for span in soup.select(_DETAIL_SELECTOR):
        label, separator, value = span.get_text("", strip=True).partition(":")
        if separator and label.strip():
            result[label.strip()] = value.strip()
    return result
