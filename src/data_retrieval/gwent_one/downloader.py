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

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import requests

from src.data_retrieval.rate_limiter import RateLimiter

_REQUEST_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class GwentOneDownloadResult:
    """Paths to each page fragment a completed fetch() wrote to disk.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    page_paths: list[Path]  # one path per page fetched, in request order


class GwentOneDownloader:
    """Downloads one or more raw HTML page fragments from gwent.one's
    card search AJAX endpoint.

    Single-consumer to src/data_retrieval/gwent_one/ — no other source
    directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/gwent_one")
    DEFAULT_AJAX_URL: ClassVar[str] = "https://gwent.one/search/ajax"

    def __init__(
        self,
        ajax_url: str | None = None,
        raw_data_dir: Path | None = None,
        *,
        rate_limiter: RateLimiter,
        language: str = "en",
    ) -> None:
        """
        Inputs:
            ajax_url: full URL to gwent.one's search AJAX endpoint.
                Defaults to DEFAULT_AJAX_URL when omitted — unlike
                Scryfall's per-dump timestamped URL or HearthstoneJSON's
                listing URL, this endpoint is a fixed address with no
                per-call variation, so capturing it as a class default
                (rather than requiring every call site to know and
                pass it) is worth doing here specifically, so this
                exact endpoint stays discoverable from the code itself
                rather than only from this session's history.
            raw_data_dir: directory each page's HTML fragment is
                written into. Defaults to DEFAULT_RAW_DATA_DIR when
                omitted (expected to be a path under data/raw, per
                src/README.md — not this class's concern to enforce,
                just to receive).
            rate_limiter: paces every outgoing request this class
                makes. Passed in rather than constructed internally
                (dependency injection — PATTERNS.md), same convention
                as HearthstoneJsonDownloader — so the same shared
                RateLimiter instance can be reused across sources, and
                so tests can supply a fast/no-op limiter.
            language: value sent as the `lang` form field (e.g. "en").
        Output: none (constructor).
        Side effects: none — no I/O happens until fetch()/
            fetch_page() are called.
        Exceptions: none.
        """
        self.ajax_url = ajax_url if ajax_url is not None else self.DEFAULT_AJAX_URL
        self.raw_data_dir = (
            raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR
        )
        self.rate_limiter = rate_limiter
        self.language = language

    def fetch(self, result_limit: int, pages: int = 1) -> GwentOneDownloadResult:
        """Entry point: fetch `pages` page(s), each capped at
        `result_limit` cards, saving each page's raw HTML fragment to
        disk.

        Composed of one fetch_page() call per page in range(1, pages +
        1) — no additional logic beyond calling that and collecting
        the resulting paths.

        Inputs:
            result_limit: value sent as the `total` form field —
                gwent.one's per-request result cap. Set above the live
                card count to get every card back in a single page
                (see module docstring).
            pages: number of pages to fetch, starting at 1. Defaults
                to 1 — today's expected usage is one page with a high
                result_limit; multi-page support exists for a future
                filtered/narrower query that might need more than one.
        Output: GwentOneDownloadResult listing each page's saved path,
            in request order.
        Side effects: one paced network request per page; creates
            raw_data_dir if missing; writes one file per page.
        Exceptions: whatever fetch_page() raises — one page failing is
            a hard failure for the whole fetch() call. Unlike
            HearthstoneJsonDownloader's per-build best-effort outcome
            union, there's no per-page independence to preserve today
            (single-page use is the only exercised case) — worth
            revisiting if multi-page fetching against a real filtered
            query becomes common enough that one bad page shouldn't
            lose the rest.

        Example:
            >>> downloader = GwentOneDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> result = downloader.fetch(result_limit=1300)
        """
        page_paths = [
            self.fetch_page(page, result_limit) for page in range(1, pages + 1)
        ]
        return GwentOneDownloadResult(page_paths=page_paths)

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
