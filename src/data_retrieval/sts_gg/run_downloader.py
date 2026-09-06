"""Downloads Slay the Spire 2 run data from sts.gg, in two phases.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job) — in particular, whatever card-alias
differences this source turns out to have against spire_codex's own
run export are explicitly out of scope here, deferred to a future
data_refinement pass.

Two phases, kept as two separate public methods rather than composed
into one fetch() — same rationale as PlayGwentDownloader
(src/data_retrieval/play_gwent/downloader.py): they're expected to be
run and re-run as separate steps against a live, paginated site.

    1. phase_1() — pages sts.gg's leaderboard API
       (GET /api/sts2/leaderboard?page=<n>&limit=<limit>) and writes
       every run id it collects to run_ids.txt.

       CONFIRMED LIVE (2026-09-05, curl against the real endpoint):
       the response shape is {"runs": [...], "total": <int>,
       "page": <int>, "totalPages": <int>} (see
       data/tmp/leaderboard_json_results.json for a saved sample).
       totalPages is reported directly on every page, so phase_1 pages
       1..totalPages rather than inferring the last page from a short
       one, unlike PlayGwentDownloader.phase_1's heuristic. Also
       confirmed live: the `limit` query param does NOT change the
       server's page size — limit=100, limit=500, and limit=1000 all
       returned ~79-82 runs per page and totalPages=10. Send it anyway
       (a harmless, possibly-someday-honored hint), but never use it to
       predict page size or loop bounds — only totalPages governs the
       walk. `total` itself fluctuates between calls (943 vs. 982
       moments apart) — this is a live, constantly-submitted
       leaderboard of roughly a thousand runs, not a fixed or bulk
       historical corpus; a run present on one walk can be at a
       different page, or briefly absent, on the next.

       Reads each page via download_to_string() (not download_to_file)
       since the body needs to be inspected (for totalPages and each
       run's id) before deciding whether to keep paging — there's
       nothing here that belongs on disk as its own file the way a run
       detail response does in phase_2().

    2. phase_2() — reads run_ids.txt and, for every run id not already
       recorded in runs_manifest.txt, fetches that run's detail JSON
       and appends it to runs.jsonl. One shared file rather than one
       file per run: at this source's actual scale (~1,000 runs on the
       live leaderboard, not spire_codex's ~1.5M-run bulk export) file
       count was never the concern — a single jsonl is simply less to
       manage than a thousand tiny files, and the manifest/jsonl
       append-order contract already has a well-established, safe
       pattern in this container (PlayGwentDownloader.phase_2): the
       data row is written before the manifest row, so a crash between
       the two leaves an orphan data row (harmless — the next run just
       re-fetches and re-appends that one run, producing a duplicate
       line a downstream reader can de-dup on) rather than the reverse
       failure mode, where the manifest would claim a run is done while
       its data row was never written. Kept uncompressed for now, same
       as PlayGwentDownloader's guides.jsonl — revisit if this source's
       corpus turns out to grow well past its current ~1,000-run scale.

       Retries are handled by download_to_string() itself (default 3
       attempts) — this class has no private retry helper of its own.
       See src/data_retrieval/TODO.md's note on migrating other
       sources' private retry loops onto that shared function too.

CONFIRMED LIVE (2026-09-05): the run detail endpoint is
GET /api/sts2/leaderboard/run/<run_id> — NOT /api/sts2/run/<run_id> (an
earlier guess in this file, before the real endpoint was tested against
the live site). data/tmp/run.json is a saved sample response.
"""

import json
from pathlib import Path
from typing import ClassVar

from tqdm import tqdm

from src.data_retrieval.download_utils import (
    append_with_manifest,
    download_to_string,
    read_manifest,
)
from src.data_retrieval.rate_limiter import RateLimiter


class STSGGRunDownloader:
    """Downloads Slay the Spire 2 run ids and run detail JSON from
    sts.gg.

    Single-consumer to src/data_retrieval/sts_gg/ — no other source
    directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/sts_gg")
    DEFAULT_LEADERBOARD_URL: ClassVar[str] = (
        "https://sts.gg/api/sts2/leaderboard?page={page}&limit={limit}"
    )
    DEFAULT_RUN_DETAIL_URL: ClassVar[str] = (
        "https://sts.gg/api/sts2/leaderboard/run/{run_id}"
    )

    def __init__(
        self,
        rate_limiter: RateLimiter,
        output_dir: Path | None = None,
        leaderboard_url: str | None = None,
        run_detail_url: str | None = None,
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
                (run_ids.txt, runs.jsonl, runs_manifest.txt). Defaults
                to DEFAULT_RAW_DATA_DIR when omitted (expected to be a
                path under data/raw, per src/README.md — not this
                class's concern to enforce, just to receive).
            leaderboard_url: URL template for the leaderboard API,
                containing "{page}" and "{limit}" placeholders.
                Defaults to DEFAULT_LEADERBOARD_URL when omitted.
            run_detail_url: URL template for one run's detail JSON,
                containing a "{run_id}" placeholder. Defaults to
                DEFAULT_RUN_DETAIL_URL when omitted (confirmed live —
                see this module's docstring).
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/phase_2()
            are called.
        Exceptions: none.
        """
        self.rate_limiter = rate_limiter
        self.output_dir = output_dir or self.DEFAULT_RAW_DATA_DIR
        self.leaderboard_url = leaderboard_url or self.DEFAULT_LEADERBOARD_URL
        self.run_detail_url = run_detail_url or self.DEFAULT_RUN_DETAIL_URL

    def phase_1(self, per_page_count: int = 100) -> Path:
        """Page through the leaderboard API and write every run id
        collected to disk.

        Reads totalPages off the first page's response and pages
        1..totalPages (see this module's docstring — totalPages is the
        only authoritative stopping condition; per_page_count/`limit`
        is sent but confirmed NOT to change the server's actual page
        size). Ids are collected across all pages, in response order,
        without deduplicating as it goes; deduplication (preserving
        first-seen order) happens once, after the loop, right before
        writing — same rationale as PlayGwentDownloader.phase_1: a
        duplicate id seen mid-loop can be a real signal the live
        leaderboard shifted rank order during the crawl, not just
        redundant data, so it's deliberately not hidden by
        deduplicating early.

        Inputs:
            per_page_count: value sent as the API's own "limit" query
                param. Confirmed live to have no effect on actual page
                size (see module docstring) — kept as a parameter in
                case that ever changes, not relied on for loop logic.
        Output: path to the written ids file (output_dir/run_ids.txt).
        Side effects: one paced network request per page (via
            download_to_string(), which retries internally); creates
            output_dir if missing; writes one file
            (output_dir/run_ids.txt), overwriting any existing one;
            prints a tqdm progress bar to stderr, sized against
            totalPages once known.
        Exceptions: raises on a page that still fails after
            download_to_string()'s retries (one page failing is a hard
            failure for the whole call; pages are a sequential walk,
            not a fixed list of independent ids, so there's no "skip
            one, keep going" available here), or on a response body
            that isn't valid JSON in the expected {"runs": [...],
            "totalPages": ...} shape.

        Example:
            >>> downloader = STSGGRunDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> ids_path = downloader.phase_1()
        """
        first_page = self._fetch_leaderboard_page(1, per_page_count)
        total_pages = first_page["totalPages"]
        collected_ids = [run["id"] for run in first_page["runs"]]

        with tqdm(
            total=total_pages, initial=1, desc="sts.gg leaderboard", unit="page"
        ) as progress:
            for page in range(2, total_pages + 1):
                page_body = self._fetch_leaderboard_page(page, per_page_count)
                collected_ids.extend(run["id"] for run in page_body["runs"])
                progress.update(1)

        deduplicated_ids = list(dict.fromkeys(collected_ids))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ids_path = self.output_dir / "run_ids.txt"
        ids_path.write_text(
            "".join(f"{run_id}\n" for run_id in deduplicated_ids),
            encoding="utf-8",
        )

        return ids_path

    def phase_2(self) -> Path:
        """For every run id not already recorded in
        runs_manifest.txt, fetch that run's detail JSON and append it
        to runs.jsonl.

        runs.jsonl is the actual data (one JSON object per line,
        appended, never rewritten); runs_manifest.txt is purely a fast
        resumability index over it, not a second source of truth. Per
        run, the data row is appended to runs.jsonl BEFORE that run's
        id is appended to runs_manifest.txt — deliberately, so a crash
        between the two leaves a data row with no matching manifest
        entry (a harmless, self-correcting gap: the next run just
        re-fetches and re-appends that one run, producing a duplicate
        line a downstream reader can de-dup on) rather than the reverse
        failure mode, where the manifest would claim a run is done while
        its data row was never written — a silent, undetectable loss.
        Same contract as PlayGwentDownloader.phase_2.

        A run that still fails after download_to_string()'s retries is
        logged via tqdm.write() and skipped, not raised — best-effort,
        same convention as PlayGwentDownloader.phase_2: a run over many
        thousands of run ids shouldn't be lost to one bad id. A skipped
        run is simply absent from runs_manifest.txt, so it's
        indistinguishable from "not yet attempted" and a later phase_2()
        run will retry it like any other.

        Inputs: none (uses self.output_dir, self.run_detail_url,
            self.rate_limiter — reads output_dir/run_ids.txt, written by
            phase_1()).
        Output: path to the JSONL data file (output_dir/runs.jsonl).
        Side effects: one paced network request (more on retry — see
            download_to_string()) per run id not already in
            runs_manifest.txt; appends one line to runs.jsonl and one
            line to runs_manifest.txt per successfully-fetched run,
            flushing each write immediately; prints a tqdm progress bar
            to stderr covering all of run_ids (skipped/already-
            downloaded ids still advance the bar, so its position
            reflects "how far through run_ids.txt," not just "how many
            new fetches happened"); prints one tqdm.write() line per run
            that fails and gets skipped.
        Exceptions: raises only if output_dir/run_ids.txt doesn't exist
            (phase_1() hasn't been run) — per-run failures are caught
            internally (see above), never propagated.

        Example:
            >>> downloader = STSGGRunDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> downloader.phase_1()
            >>> runs_path = downloader.phase_2()
        """
        ids_path = self.output_dir / "run_ids.txt"
        run_ids = [
            line
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        manifest_path = self.output_dir / "runs_manifest.txt"
        already_downloaded = read_manifest(manifest_path)

        runs_path = self.output_dir / "runs.jsonl"

        for run_id in tqdm(run_ids, desc="sts.gg run details", unit="run"):
            if run_id in already_downloaded:
                continue

            self.rate_limiter.wait()
            try:
                url = self.run_detail_url.format(run_id=run_id)
                run_payload = json.loads(download_to_string(url))
            except Exception as error:
                # Best-effort, same convention as PlayGwentDownloader.phase_2:
                # one run still failing after download_to_string()'s own
                # retries shouldn't lose the rest of a run over many
                # thousands of ids. A skipped run is simply absent from
                # runs_manifest.txt, so it's indistinguishable from "not
                # yet attempted" and a later phase_2() call retries it
                # like any other.
                tqdm.write(f"sts.gg run {run_id} failed: {error}")
                continue

            append_with_manifest(
                runs_path, manifest_path, json.dumps(run_payload), run_id
            )

        return runs_path

    def _fetch_leaderboard_page(self, page: int, per_page_count: int) -> dict:
        """Fetch one leaderboard page and return its parsed JSON body.

        Private helper — single consumer is phase_1(). Uses
        download_to_string() (retries handled there) rather than
        download_to_file(): the body needs to be inspected (totalPages,
        each run's id) before deciding whether to keep paging, and a
        leaderboard page isn't itself one of this class's permanent
        outputs the way run.jsonl's rows are — nothing here belongs on
        disk as its own file.

        Inputs:
            page: 1-indexed page number.
            per_page_count: value sent as the "limit" query param (see
                phase_1()'s docstring — confirmed not to affect actual
                page size, sent anyway).
        Output: the page's parsed JSON body ({"runs": [...], "total":
            ..., "page": ..., "totalPages": ...}).
        Side effects: one paced network request.
        Exceptions: raises whatever download_to_string() raises once
            its retries are exhausted, or if the response body isn't
            valid JSON.
        """
        self.rate_limiter.wait()
        url = self.leaderboard_url.format(page=page, limit=per_page_count)
        return json.loads(download_to_string(url))
