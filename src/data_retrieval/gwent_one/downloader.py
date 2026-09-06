"""Downloads Gwent card listings from gwent.one's search AJAX endpoint.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job) — in particular, this class saves each page's
raw HTML fragment untouched; extracting per-card fields out of the
`card-wrap card-data` divs is a future GwentOneCardIngestionStage's
job, not this downloader's.

gwent.one has no bulk data dump or public DB export. The one endpoint
that works: POST https://gwent.one/search/ajax, form-encoded body
`total=<result_limit>&lang=<language>&page=<page>` — `total` is a
per-request result cap, not a fixed page size (the site's own JS just
sets it low, e.g. 140, for slow devices; the server returns however
many are asked for). `page` is a real pagination axis, kept here for a
future filtered/narrower query (by faction/set/etc.) that might need
more than one page — today's typical call uses a single page with
result_limit set above the live card count (visible in gwent.one's own
"Cards: N" footer), returning every card in one response.
"""

from pathlib import Path
from typing import ClassVar

import requests

from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_REQUEST_TIMEOUT_SECONDS = 60


class GwentOneDownloader(Downloader):
    """Downloads one or more raw HTML page fragments from gwent.one's
    card search AJAX endpoint.

    Single-consumer to src/data_retrieval/gwent_one/ — no other source
    directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/gwent_one")
    DEFAULT_AJAX_URL: ClassVar[str] = "https://gwent.one/search/ajax"

    def __init__(
        self,
        result_limit: int,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        *,
        pages: int = 1,
        ajax_url: str | None = None,
        language: str = "en",
    ) -> None:
        """
        Inputs:
            result_limit: value sent as the `total` form field —
                gwent.one's per-request result cap. Set above the live
                card count to get every card back in a single page
                (see module docstring). Required and positional: no
                sensible source-wide default (it depends on gwent.one's
                current live card count).
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
            pages: number of pages phase_1() fetches, starting at 1.
                Defaults to 1 — today's expected usage is one page with
                a high result_limit; multi-page support exists for a
                future filtered/narrower query that might need more
                than one. Moved here from a phase_1()-call argument
                (this class's old fetch() method) since
                Downloader.phase_1() takes no arguments.
            ajax_url: full URL to gwent.one's search AJAX endpoint.
                Defaults to DEFAULT_AJAX_URL when omitted — unlike
                Scryfall's per-dump timestamped URL or HearthstoneJSON's
                listing URL, this endpoint is a fixed address with no
                per-call variation, so capturing it as a class default
                (rather than requiring every call site to know and
                pass it) is worth doing here specifically, so this
                exact endpoint stays discoverable from the code itself
                rather than only from this session's history.
            language: value sent as the `lang` form field (e.g. "en").
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/
            fetch_page() are called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.result_limit = result_limit
        self.pages = pages
        self.ajax_url = ajax_url if ajax_url is not None else self.DEFAULT_AJAX_URL
        self.language = language

    def _run_phase_1(self) -> Path:
        """Fetch self.pages page(s), each capped at self.result_limit
        cards, saving each page's raw HTML fragment to disk, and
        return raw_data_dir — one page per file, with no single
        artifact that's "the" result when self.pages > 1.

        Composed of one fetch_page() call per page in range(1,
        self.pages + 1) — no additional logic beyond calling that.
        The full page-path list stays available via calling
        fetch_page() directly.

        Inputs: none (uses self.result_limit, self.pages,
            self.raw_data_dir, self.rate_limiter).
        Output: self.raw_data_dir.
        Side effects: one paced network request per page; creates
            raw_data_dir if missing; writes one file per page.
        Exceptions: whatever fetch_page() raises — one page failing is
            a hard failure for the whole call. Unlike
            HearthstoneJsonDownloader's per-build best-effort outcome
            union, there's no per-page independence to preserve today
            (single-page use is the only exercised case) — worth
            revisiting if multi-page fetching against a real filtered
            query becomes common enough that one bad page shouldn't
            lose the rest.

        Example:
            >>> downloader = GwentOneDownloader(
            ...     result_limit=1300,
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> raw_data_dir = downloader.phase_1()
        """
        for page in range(1, self.pages + 1):
            self.fetch_page(page, self.result_limit)
        return self.raw_data_dir

    def fetch_page(self, page: int, result_limit: int) -> Path:
        """Fetch one page from the AJAX endpoint and save its raw HTML
        response to raw_data_dir, always overwriting any existing file
        for this page.

        Always-overwrite (no skip-if-exists) is a deliberate departure
        from HearthstoneJsonDownloader's resumability: that source's
        per-build files are immutable history, so skipping an
        already-downloaded build is safe. gwent.one's listing reflects
        the *current* card pool (new sets get added over time), so a
        stale on-disk page would silently go unrefreshed if this
        skipped existing files the way that downloader does.

        Inputs:
            page: 1-indexed page number, sent as the `page` form
                field.
            result_limit: value sent as the `total` form field.
        Output: path to the saved HTML fragment
                (raw_data_dir/page_<page>.html).
        Side effects: one paced network request; creates raw_data_dir
            if missing; writes one file to raw_data_dir. Any
            partially-written file is removed before the exception
            propagates, same convention as the sibling downloaders.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file.
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        destination_path = self.raw_data_dir / f"page_{page}.html"

        self.rate_limiter.wait()
        response = requests.post(
            self.ajax_url,
            data={"total": result_limit, "lang": self.language, "page": page},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        try:
            destination_path.write_text(response.text, encoding="utf-8")
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise

        return destination_path
