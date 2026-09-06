"""Downloads spire-codex's bulk run export in cursor-paginated pages.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

spire-codex's GET /api/exports/runs serves every submitted Slay the
Spire 2 run as gzipped JSONL (one full raw run per line — players,
map_point_history, acts, deck, relics, card_choices), ordered by
(submitted_at, _id). Unlike SpireCodexCardDownloader's single static
file, this endpoint is a paginated walk:

- `limit` bounds a page to at most that many runs (server max 50000).
  Omit for the full, unbounded corpus in one page — not used here,
  since _run_phase_1() always pages explicitly so a slow full-corpus pull can
  resume after being interrupted (per manual testing: the full export
  is very slow).
- The response body for one page is itself gzip-compressed. The next
  page's cursor is NOT in the body — it's the `X-Next-Cursor` response
  header. Its absence means this was the last page. This is NOT the
  same as "this page had fewer than `limit` lines": a page's line
  count can legitimately differ from `limit` (a multiplayer run emits
  one raw line per player; missing/unreadable run files are skipped
  server-side) — only the missing header means "done".
- `start`/`end` are an optional half-open [start, end) window on
  submitted_at (ISO-8601). Per the API docs, these must stay constant
  across a paged sequence — the cursor doesn't embed the window itself.

Resumability: each page is written to its own gzip file rather than
one accumulated file, mirroring HearthstoneJsonDownloader's per-file
skip-if-exists. But unlike that source's independent per-build files,
run pages form a dependent chain — page N+1's cursor only exists once
page N has been fetched — so resuming isn't a plain existence check
per page: _run_phase_1() also needs to know the cursor a resumed walk should
continue from. That cursor is persisted next to each page as a small
sidecar file (see _NEXT_CURSOR_SIDECAR_TEMPLATE), containing the
X-Next-Cursor header value or an empty string for the corpus's last
page. The sidecar is written only after that page's data file is
fully written, so a page file present without its sidecar means a
prior run was interrupted mid-page and that page is re-fetched rather
than trusted. All on-disk page state is read through a single scan
(_scan_downloaded_pages()) so nothing else re-derives it independently
and risks disagreeing with that scan.

A single (start, end) window is recorded in a small metadata file (see
_WINDOW_METADATA_FILENAME) the first time _run_phase_1() is called against a
given raw_data_dir, and every later _run_phase_1() call against that same
raw_data_dir is checked against it — a caller resuming with a
different window than the original walk used gets a raised error
rather than a silently spliced result. This check is keyed on the
window file's own presence, not on whether any page has completed yet
— a walk that recorded its window but failed before finishing page 0
must still refuse a differently-windowed retry, not treat itself as
fresh again.

Every outgoing request (each page, plus the one-off stats call below)
is paced through a RateLimiter — confirmed necessary by hitting a live
429 during manual testing (~120 rapid sequential requests), despite no
documented rate limit for this endpoint. Per this project's own
politeness default, any external source gets paced to ~1 request/sec
absent a source-specific reason to differ.

Progress reporting: the export endpoint itself gives no total/
remaining-page count (only presence/absence of X-Next-Cursor — see
above), so _run_phase_1() separately calls GET /api/runs/stats once, purely
for its `total_runs` field, to seed a tqdm progress bar with an
*approximate* page count (ceil(total_runs / page_limit)). Confirmed
live: /api/runs/list's own `total`/`total_pages` fields are capped at
10000/200 (a UI pagination cap, not the true corpus size) and are NOT
usable for this; /api/runs/stats's total_runs (~1.5M at the time this
was checked) is not capped. This estimate is intentionally
approximate, not authoritative: total_runs counts runs, while the
export emits one line per player (so multiplayer runs push the true
line/page count up), and the export only includes "official
characters" (so it could push the true count down). This call is
best-effort and purely cosmetic — if it fails, _run_phase_1() falls back to
an indeterminate progress bar rather than aborting the download.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import requests
from tqdm import tqdm

from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_REQUEST_TIMEOUT_SECONDS = 60
_DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB, matches download_to_file
_NEXT_CURSOR_HEADER = "X-Next-Cursor"
_PAGE_FILENAME_TEMPLATE = "page_{page_index:05d}.jsonl.gz"
_NEXT_CURSOR_SIDECAR_TEMPLATE = "page_{page_index:05d}.next_cursor"
_WINDOW_METADATA_FILENAME = "window.json"


@dataclass(frozen=True)
class _FetchedPage:
    """One page's on-disk result plus the cursor for the page after it.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    path: Path
    next_cursor: str | None  # None means this was the corpus's last page.


@dataclass(frozen=True)
class _PagesRemaining:
    """Resume point: at least one more page needs fetching.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    next_page_index: int
    cursor: str | None  # None only when next_page_index == 0 (fresh walk).


@dataclass(frozen=True)
class _ExportComplete:
    """Resume point: every page of this corpus/window is already on disk.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """


# Discriminated union (PRINCIPLES.md "illegal states unrepresentable",
# same idiom as HearthstoneJsonDownloader's BuildDownloadOutcome):
# resuming a paged walk is either "continue from here" or "nothing left
# to do", never an ambiguous index-plus-maybe-cursor pair.
_ResumePoint = _PagesRemaining | _ExportComplete


class SpireCodexRunDownloader(Downloader):
    """Downloads spire-codex's full run export, one gzip file per page.

    Single-consumer to src/data_retrieval/spire_codex/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/spire_codex/runs")
    DEFAULT_EXPORT_URL: ClassVar[str] = "https://spire-codex.com/api/exports/runs"
    DEFAULT_STATS_URL: ClassVar[str] = "https://spire-codex.com/api/runs/stats"
    DEFAULT_PAGE_LIMIT: ClassVar[int] = 50000  # server-documented max.

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        *,
        export_url: str | None = None,
        stats_url: str | None = None,
        page_limit: int | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
            export_url: URL of the runs export endpoint. Defaults to
                DEFAULT_EXPORT_URL when omitted.
            stats_url: URL of the community-stats endpoint, queried
                only to approximate a total page count for phase_1()'s
                progress bar (see this module's docstring). Defaults
                to DEFAULT_STATS_URL when omitted.
            page_limit: max runs per page, passed as the `limit` query
                param on every request. Defaults to DEFAULT_PAGE_LIMIT
                (the server's documented max) when omitted, so a full
                walk takes as few requests as the server allows.
            start: inclusive lower bound on submitted_at (ISO-8601),
                passed through to every page request. Held constant
                across a resumed walk — a resumed call whose start
                doesn't match the walk's recorded window raises (see
                phase_1()'s Exceptions). A constructor argument rather
                than a phase_1() call argument specifically because it
                must never vary across calls against the same
                raw_data_dir — see _record_window().
            end: exclusive upper bound on submitted_at (ISO-8601), same
                constancy requirement and mismatch behavior as start.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1() is called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.export_url = (
            export_url if export_url is not None else self.DEFAULT_EXPORT_URL
        )
        self.stats_url = stats_url if stats_url is not None else self.DEFAULT_STATS_URL
        self.page_limit = (
            page_limit if page_limit is not None else self.DEFAULT_PAGE_LIMIT
        )
        self.start = start
        self.end = end

    def _run_phase_1(self) -> Path:
        """Walk every page of the run export, writing one gzip file per
        page to raw_data_dir, resuming from the last page already on
        disk for this raw_data_dir, and return raw_data_dir — one page
        per file, with no single artifact that's "the" result.

        Inputs: none (uses self.start, self.end, self.raw_data_dir,
            self.rate_limiter). The full per-page path list stays
            available via calling _scan_downloaded_pages() directly.
        Output: self.raw_data_dir.
        Side effects: creates raw_data_dir if missing; one best-effort,
            rate-limited network request to stats_url for the progress
            bar's total (never raises — see _estimate_total_pages); one
            rate-limited network request per page fetched this call
            (already-complete pages cost no request); writes one page
            file plus one cursor sidecar file per page fetched this
            call; writes this walk's window metadata file once, the
            first time phase_1() is ever called against this
            raw_data_dir; prints a tqdm progress bar to stderr covering
            the walk.
        Exceptions: raises ValueError if raw_data_dir already holds
            pages from a walk whose recorded (start, end) window
            doesn't match self.start/self.end. Raises on network
            failure (e.g. connection error, non-2xx response) or on
            failure to write a file. A paged walk is a dependent chain
            (page N+1 needs page N's cursor), so unlike
            HearthstoneJsonDownloader this is not best-effort: one page
            failing stops the whole call, and a later phase_1() call
            resumes from the last complete page rather than restarting.

        Example:
            >>> downloader = SpireCodexRunDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=60),
            ... )
            >>> raw_data_dir = downloader.phase_1()
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Record this walk's window the first time it's ever called
        # against raw_data_dir, or verify a later call still matches it
        # — keyed on the window file's own presence, not on whether any
        # page has completed yet (see module docstring).
        self._record_window(self.start, self.end)

        # Single scan of on-disk state — both the pages already present
        # and this walk's resume point are derived from it, so they
        # can't disagree with each other (see module docstring).
        downloaded_pages = self._scan_downloaded_pages()

        resume_point = self._resume_point(downloaded_pages)
        if isinstance(resume_point, _ExportComplete):
            return self.raw_data_dir

        page_index = resume_point.next_page_index
        cursor = resume_point.cursor
        total_pages_estimate = self._estimate_total_pages()

        # Walk forward one page at a time until the server stops
        # returning a next cursor. total_pages_estimate is a rough
        # approximation (see module docstring) — tqdm handles it going
        # over 100% or being None (indeterminate) without failing.
        with tqdm(
            total=total_pages_estimate,
            initial=len(downloaded_pages),
            desc="spire-codex runs export",
            unit="page",
        ) as progress:
            while True:
                page = self._fetch_page(
                    page_index, cursor, start=self.start, end=self.end
                )
                progress.update(1)

                if page.next_cursor is None:
                    break

                cursor = page.next_cursor
                page_index += 1

        return self.raw_data_dir

    def _scan_downloaded_pages(self) -> list[_FetchedPage]:
        """Read every complete page already in raw_data_dir, in page
        order — the single source of truth for this walk's on-disk
        state.

        Private helper — single consumer is _run_phase_1(). A page counts as
        complete only when both its data file and its cursor sidecar
        are present (see this module's docstring on why the sidecar is
        written second) — an incomplete trailing page is treated as
        not present, so _run_phase_1() re-fetches it. Pages are a contiguous
        sequence from index 0 (each one's cursor is only knowable after
        fetching the one before it), so the scan stops at the first
        gap rather than needing to know an upper bound up front.

        Inputs: none (uses self.raw_data_dir).
        Output: complete pages, ascending by page index, each carrying
            the cursor its sidecar recorded (None if that sidecar's
            content was empty, i.e. that page was the corpus's last).
        Side effects: reads raw_data_dir's contents.
        Exceptions: none expected beyond filesystem errors reading
            raw_data_dir.
        """
        downloaded_pages = []
        page_index = 0

        while True:
            page_path = self.raw_data_dir / _PAGE_FILENAME_TEMPLATE.format(
                page_index=page_index
            )
            cursor_sidecar_path = (
                self.raw_data_dir
                / _NEXT_CURSOR_SIDECAR_TEMPLATE.format(page_index=page_index)
            )
            if not (page_path.exists() and cursor_sidecar_path.exists()):
                break

            cursor_content = cursor_sidecar_path.read_text(encoding="utf-8")
            next_cursor = cursor_content if cursor_content else None
            downloaded_pages.append(
                _FetchedPage(path=page_path, next_cursor=next_cursor)
            )
            page_index += 1

        return downloaded_pages

    def _resume_point(self, downloaded_pages: list[_FetchedPage]) -> _ResumePoint:
        """Derive where a walk should resume from an already-scanned
        page list.

        Private helper — single consumer is _run_phase_1(). Pure: takes
        _scan_downloaded_pages()'s own output rather than re-reading
        raw_data_dir itself, so this can't drift from what _run_phase_1()
        already knows is on disk.

        Inputs:
            downloaded_pages: this walk's complete pages, as returned
                by _scan_downloaded_pages().
        Output: _PagesRemaining naming the next page index and cursor
            to fetch, or _ExportComplete if downloaded_pages is
            non-empty and its last entry's next_cursor is None.
        Side effects: none.
        Exceptions: none.
        """
        if not downloaded_pages:
            return _PagesRemaining(next_page_index=0, cursor=None)

        last_page = downloaded_pages[-1]
        if last_page.next_cursor is None:
            return _ExportComplete()

        return _PagesRemaining(
            next_page_index=len(downloaded_pages), cursor=last_page.next_cursor
        )

    def _record_window(self, start: str | None, end: str | None) -> None:
        """Record (start, end) as this raw_data_dir's walk window if
        none is recorded yet; otherwise verify this call's (start, end)
        matches what's already recorded.

        Private helper — single consumer is _run_phase_1(), called
        unconditionally on every call, before any page work — so the
        window file's own presence (not page completeness) is what
        decides get-vs-set, closing the gap a page-completeness-gated
        check would leave open (a walk that recorded its window but
        failed before finishing page 0 must still refuse a
        differently-windowed retry).

        Inputs:
            start: this call's start argument.
            end: this call's end argument.
        Output: none.
        Side effects: reads this raw_data_dir's window metadata file;
            writes it if it doesn't exist yet.
        Exceptions: raises ValueError if a window metadata file already
            exists and doesn't match (start, end). Raises on failure to
            write the file.
        """
        window_path = self.raw_data_dir / _WINDOW_METADATA_FILENAME
        requested_window = {"start": start, "end": end}

        if window_path.exists():
            recorded_window = json.loads(window_path.read_text(encoding="utf-8"))
            if recorded_window != requested_window:
                raise ValueError(
                    f"{self.raw_data_dir} already holds a run export walk for window "
                    f"{recorded_window}, which does not match the requested window "
                    f"{requested_window}. Use a different raw_data_dir for a different "
                    "window."
                )
            return

        window_path.write_text(json.dumps(requested_window), encoding="utf-8")

    def _estimate_total_pages(self) -> int | None:
        """Best-effort, approximate total page count for _run_phase_1()'s
        progress bar.

        Private helper — single consumer is _run_phase_1(). Purely cosmetic:
        this is never allowed to fail _run_phase_1() itself, since the actual
        download doesn't depend on knowing a total in advance (see
        this module's docstring for why the estimate can run either
        high or low).

        Inputs: none (uses self.stats_url, self.page_limit,
            self.rate_limiter).
        Output: ceil(total_runs / page_limit) from stats_url's
            response, or None if the request fails, times out, or the
            response doesn't contain a usable total_runs value.
        Side effects: one paced network request to stats_url.
        Exceptions: none — every failure mode returns None instead of
            raising.
        """
        try:
            self.rate_limiter.wait()
            response = requests.get(self.stats_url, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            total_runs = response.json()["total_runs"]
            return math.ceil(total_runs / self.page_limit)
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
            ZeroDivisionError,
        ):
            return None

    def _fetch_page(
        self, page_index: int, cursor: str | None, *, start: str | None, end: str | None
    ) -> _FetchedPage:
        """Download one page of the run export and persist it plus its
        next-cursor sidecar.

        Private helper — single consumer is _run_phase_1(). Not built on
        download_to_file: that helper's contract is "URL in, whole
        file on disk, no response metadata out" — this needs the
        X-Next-Cursor response header before/while streaming the body,
        which download_to_file has no way to surface.

        Inputs:
            page_index: which page this is, used to name both the page
                file and its sidecar.
            cursor: opaque cursor to request this page with, or None
                for the first page of a fresh walk.
            start: passed through as the `start` query param.
            end: passed through as the `end` query param.
        Output: the page's on-disk path plus the cursor for the
            following page (None if this was the last page).
        Side effects: one paced network request; writes the page's
            gzip file and its cursor sidecar file. Any partially-
            written page file is removed before the exception
            propagates (same convention as download_to_file).
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write either file.
        """
        params = self._page_request_params(cursor=cursor, start=start, end=end)

        page_path = self.raw_data_dir / _PAGE_FILENAME_TEMPLATE.format(
            page_index=page_index
        )
        cursor_sidecar_path = self.raw_data_dir / _NEXT_CURSOR_SIDECAR_TEMPLATE.format(
            page_index=page_index
        )

        self.rate_limiter.wait()
        try:
            with requests.get(
                self.export_url,
                params=params,
                stream=True,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                next_cursor = response.headers.get(_NEXT_CURSOR_HEADER)
                with open(page_path, "wb") as page_file:
                    for chunk in response.iter_content(
                        chunk_size=_DEFAULT_CHUNK_SIZE_BYTES
                    ):
                        page_file.write(chunk)
        except Exception:
            page_path.unlink(missing_ok=True)
            raise

        cursor_sidecar_path.write_text(next_cursor or "", encoding="utf-8")
        return _FetchedPage(path=page_path, next_cursor=next_cursor)

    def _page_request_params(
        self, *, cursor: str | None, start: str | None, end: str | None
    ) -> dict[str, str | int]:
        """Build the export endpoint's query params for one page request.

        Private helper — single consumer is _fetch_page().

        Inputs: cursor, start, end - each included only if not None.
        Output: dict, always containing "limit" (self.page_limit).
        Side effects: none.
        Exceptions: none.
        """
        params: dict[str, str | int] = {"limit": self.page_limit}
        if cursor is not None:
            params["cursor"] = cursor
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        return params
