"""Downloads deck guides from playgwent.com.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job) — the only transformation this module performs
is unwrapping the HTML container playgwent.com wraps its own
already-fully-formed JSON payload in, not reshaping that payload's
contents.

Two phases, kept as two separate public methods rather than composed
into one fetch(), since they're expected to be run and re-run as
separate steps against a live, paginated, mutating site:
    1. phase_1() — pages through the guides-list API (offset/limit)
       and writes every guide id it collects to deck_ids.txt. Only the
       ids are kept — the list endpoint's per-guide summary metadata
       (author, faction, votes, ...) is discarded, a deliberate
       simplification: the real payload worth keeping lives behind
       phase_2()'s per-guide detail page, which is far richer than
       this summary endpoint anyway.
    2. phase_2() — reads deck_ids.txt and, for every id not already
       recorded in guides_manifest.txt, fetches that guide's detail
       page and appends its extracted JSON payload to guides.jsonl.

See this directory's TODO.md for a known naming inconsistency
(phase_1/phase_2, positional rate_limiter) deliberately left
unresolved for now.
"""

import html
import json
import time
from pathlib import Path
from typing import ClassVar

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.data_retrieval.download_utils import append_with_manifest, read_manifest
from src.data_retrieval.rate_limiter import RateLimiter

_REQUEST_TIMEOUT_SECONDS = 60
_MAX_FETCH_ATTEMPTS = 3  # 1 initial try + up to 2 retries
_RETRY_BACKOFF_SECONDS = 2.0
# phase_1() doesn't know the true guide count up front (that's the
# whole reason it pages until a short page signals the end) — this is
# a rough, hand-set estimate used only to size the tqdm progress bar's
# denominator so it reads as "N% of roughly this many," never used for
# pagination logic itself. A live run overshooting or undershooting it
# just makes tqdm's bar go past 100% or finish early; both are fine.
_ESTIMATED_TOTAL_GUIDES = 70_000
# The literal substring "data-state" appears dozens of times per page
# (mostly as an unquoted CSS attribute selector inside <style> blocks
# from a tippy.js stylesheet, e.g. `[data-state=visible]`) — matching
# on the substring alone, or even a quoted-attribute regex, risks
# grabbing the wrong one if a real quoted `data-state="..."` HTML
# attribute is ever added elsewhere on the page (e.g. by some other
# UI widget). Scoping the lookup to the specific
# div.wrapper > div.content > div#root element this payload is known
# to be attached to avoids that ambiguity entirely.
_ROOT_DIV_SELECTOR = "div.wrapper > div.content > div#root"


class PlayGwentDownloader:
    """Downloads deck guide ids and deck guide detail pages from
    playgwent.com.

    Single-consumer to src/data_retrieval/play_gwent/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/play_gwent")
    DEFAULT_DECK_ID_GETTER_URL: ClassVar[str] = (
        "https://www.playgwent.com/en/decks/api/guides/offset/{offset}/limit/{limit}"
    )
    DEFAULT_DECK_DETAIL_URL: ClassVar[str] = (
        "https://www.playgwent.com/en/decks/guides/{guide_id}"
    )

    def __init__(
        self,
        rate_limiter: RateLimiter,
        output_dir: Path | None = None,
        deck_id_getter_url: str | None = None,
        deck_detail_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: paces every outgoing request this class
                makes. Passed in rather than constructed internally
                (dependency injection — PATTERNS.md), same convention
                as the sibling downloaders — so the same shared
                RateLimiter instance can be reused across sources, and
                so tests can supply a fast/no-op limiter. Required and
                positional here rather than keyword-only, unlike the
                sibling downloaders — see this directory's TODO.md.
            output_dir: directory this class's output is written into
                (deck_ids.txt, guides.jsonl, guides_manifest.txt).
                Defaults to DEFAULT_RAW_DATA_DIR when omitted (expected
                to be a path under data/raw, per src/README.md — not
                this class's concern to enforce, just to receive).
            deck_id_getter_url: URL template for the guides-list API,
                containing "{offset}" and "{limit}" placeholders.
                Defaults to DEFAULT_DECK_ID_GETTER_URL when omitted.
            deck_detail_url: URL template for one guide's HTML detail
                page, containing a "{guide_id}" placeholder. Defaults
                to DEFAULT_DECK_DETAIL_URL when omitted.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/phase_2()
            are called.
        Exceptions: none.
        """
        self.rate_limiter = rate_limiter
        self.output_dir = output_dir or self.DEFAULT_RAW_DATA_DIR
        self.deck_id_getter_url = deck_id_getter_url or self.DEFAULT_DECK_ID_GETTER_URL
        self.deck_detail_url = deck_detail_url or self.DEFAULT_DECK_DETAIL_URL

    def phase_1(self, initial_offset: int = 0, per_page_count: int = 500) -> Path:
        """Page through the guides-list API and write every guide id
        collected to disk.

        Pages starting at offset=initial_offset, advancing by
        per_page_count each request, until a page's "guides" list has
        fewer than per_page_count entries (an empty list included) —
        the site's signal that it was the last page. Ids are collected
        across all pages, in response order, without deduplicating as
        it goes; deduplication (preserving first-seen order) happens
        once, after the loop, right before writing — a duplicate id
        seen mid-loop can be a real signal that the live guide list
        shifted during the crawl (a new guide inserted between two of
        this method's own requests), not just redundant data, so it's
        deliberately not hidden by deduplicating early.

        Inputs:
            initial_offset: offset to start paging from.
            per_page_count: number of guide references requested per
                page (the API's own "limit" path segment), and the
                threshold used to detect the last page.
        Output: path to the written ids file (output_dir/deck_ids.txt).
        Side effects: one paced network request per page; creates
            output_dir if missing; writes one file
            (output_dir/deck_ids.txt), overwriting any existing one;
            prints a tqdm progress bar to stderr, sized against
            _ESTIMATED_TOTAL_GUIDES (a rough guess, not the pagination
            loop's actual stopping condition — see that constant's
            comment).
        Exceptions: raises on a page that still fails after
            _MAX_FETCH_ATTEMPTS attempts (see _get_with_retries — one
            page failing this many times in a row is a hard failure
            for the whole call, unlike phase_2()'s per-guide
            best-effort skip; pages are a sequential walk, not a fixed
            list of independent ids, so there's no "skip one, keep
            going" available here), or on a response body that isn't
            valid JSON in the expected {"guides": [...]} shape.

        Example:
            >>> downloader = PlayGwentDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> ids_path = downloader.phase_1()
        """
        collected_ids: list[int] = []
        offset = initial_offset

        with tqdm(
            total=_ESTIMATED_TOTAL_GUIDES, desc="Play Gwent guide ids", unit="guide"
        ) as progress:
            while True:
                url = self.deck_id_getter_url.format(
                    offset=offset, limit=per_page_count
                )
                response = self._get_with_retries(url)

                page_guides = response.json()["guides"]
                collected_ids.extend(guide["id"] for guide in page_guides)
                progress.update(len(page_guides))

                if len(page_guides) < per_page_count:
                    break
                offset += per_page_count

        deduplicated_ids = list(dict.fromkeys(collected_ids))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ids_path = self.output_dir / "deck_ids.txt"
        ids_path.write_text(
            "".join(f"{guide_id}\n" for guide_id in deduplicated_ids),
            encoding="utf-8",
        )

        return ids_path

    def phase_2(self) -> Path:
        """For every guide id not already recorded in
        guides_manifest.txt, fetch that guide's detail page and append
        its extracted JSON payload to guides.jsonl.

        guides.jsonl is the actual data (one JSON object per line,
        appended, never rewritten); guides_manifest.txt is purely a
        fast resumability index over it, not a second source of truth.
        Per guide, the data row is appended to guides.jsonl BEFORE that
        guide's id is appended to guides_manifest.txt — deliberately,
        so a crash between the two leaves a data row with no matching
        manifest entry (a harmless, self-correcting gap: the next run
        just re-fetches and re-appends that one guide, producing a
        duplicate line a downstream reader can de-dup on) rather than
        the reverse failure mode, where the manifest would claim a
        guide is done while its data row was never written — a silent,
        undetectable loss.

        A guide that still fails after _get_with_retries's retries (or
        whose page has a malformed/missing data-state payload — see
        _extract_guide_payload) is logged via tqdm.write() and skipped,
        not raised — best-effort, same convention and rationale as
        HearthstoneJsonDownloader.download_missing_builds: a run over
        tens of thousands of guides shouldn't be lost to one bad id. A
        skipped guide is simply absent from guides_manifest.txt, so
        it's indistinguishable from "not yet attempted" and a later
        phase_2() run will retry it like any other.

        Inputs: none (uses self.output_dir, self.deck_detail_url,
            self.rate_limiter — reads output_dir/deck_ids.txt, written
            by phase_1()).
        Output: path to the JSONL data file (output_dir/guides.jsonl).
        Side effects: one paced network request (more on retry — see
            _get_with_retries) per guide id not already in
            guides_manifest.txt; appends one line to guides.jsonl and
            one line to guides_manifest.txt per successfully-fetched
            guide, flushing each write immediately; prints a tqdm
            progress bar to stderr covering all of guide_ids (skipped/
            already-downloaded ids still advance the bar, so its
            position reflects "how far through deck_ids.txt," not just
            "how many new fetches happened"); prints one tqdm.write()
            line per guide that fails and gets skipped.
        Exceptions: raises only if output_dir/deck_ids.txt doesn't
            exist (phase_1() hasn't been run) — per-guide failures are
            caught internally (see above), never propagated.

        Example:
            >>> downloader = PlayGwentDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> downloader.phase_1()
            >>> guides_path = downloader.phase_2()
        """
        ids_path = self.output_dir / "deck_ids.txt"
        guide_ids = [
            int(line)
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        manifest_path = self.output_dir / "guides_manifest.txt"
        already_downloaded = {int(line) for line in read_manifest(manifest_path)}

        guides_path = self.output_dir / "guides.jsonl"

        for guide_id in tqdm(guide_ids, desc="Play Gwent guide details", unit="guide"):
            if guide_id in already_downloaded:
                continue

            try:
                url = self.deck_detail_url.format(guide_id=guide_id)
                response = self._get_with_retries(url)
                guide_payload = self._extract_guide_payload(response.text)
            except Exception as error:
                # Best-effort, same convention as
                # HearthstoneJsonDownloader.download_missing_builds: one
                # guide that's still broken after _get_with_retries's
                # retries (or has a malformed data-state payload)
                # shouldn't lose the rest of a run that can be hours
                # long. It's simply never written to guides.jsonl or
                # guides_manifest.txt, so a later phase_2() run will
                # attempt it again like any other not-yet-downloaded id
                # — the manifest's "done" meaning stays intact.
                tqdm.write(f"Play Gwent guide {guide_id} failed: {error}")
                continue

            append_with_manifest(
                guides_path, manifest_path, json.dumps(guide_payload), str(guide_id)
            )

        return guides_path

    def _get_with_retries(self, url: str) -> requests.Response:
        """GET url, retrying on failure up to _MAX_FETCH_ATTEMPTS times
        total, with a fixed _RETRY_BACKOFF_SECONDS pause between
        attempts.

        Private helper — shared by phase_1() (per-page requests) and
        phase_2() (per-guide requests), the two places this class
        makes an outgoing request; centralizing retry behavior here
        means both get it identically rather than two copies of the
        same loop drifting apart (PRINCIPLES.md section 2). This
        method itself has no opinion on what a caller does once
        retries are exhausted — it always raises — phase_1() lets that
        propagate as a hard failure, phase_2() catches it per-guide
        for its best-effort skip (see that method's docstring for why
        the two differ).

        Inputs:
            url: full URL to GET.
        Output: the successful requests.Response (2xx status).
        Side effects: one paced network request per attempt (up to
            _MAX_FETCH_ATTEMPTS); sleeps _RETRY_BACKOFF_SECONDS between
            attempts (not after the last one).
        Exceptions: raises the last attempt's requests.RequestException
            (e.g. connection error, timeout, non-2xx response) once
            _MAX_FETCH_ATTEMPTS attempts have all failed.
        """
        last_error: requests.RequestException | None = None

        for attempt in range(_MAX_FETCH_ATTEMPTS):
            self.rate_limiter.wait()
            try:
                response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt < _MAX_FETCH_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF_SECONDS)

        assert last_error is not None  # loop always runs >= 1 iteration
        raise last_error

    def _extract_guide_payload(self, html_text: str) -> dict:
        """Extract the "guide" payload embedded in a guide detail page.

        Private helper — single consumer is phase_2(), split out so
        the extraction logic is independently testable against saved
        example HTML without mocking network calls (same rationale as
        HearthstoneJsonDownloader's
        _parse_build_ids_with_enus_subdirectory split).

        The page is a server-rendered single-page-app shell; the
        entire guide payload is embedded as an HTML-escaped JSON blob
        in a `data-state` attribute on `<div id="root">`. Parsed with
        BeautifulSoup (html.parser backend, matching this project's
        existing convention — see
        data_refinement/card_binder/gwent_one/ingestion_stage.py) and
        located via _ROOT_DIV_SELECTOR rather than a raw string/regex
        search on the page text: the literal substring "data-state"
        appears dozens of times per page (mostly as an unquoted CSS
        attribute selector inside <style> blocks, unrelated to this
        payload), so scoping the lookup to the specific element this
        attribute is known to live on is what actually guarantees the
        right one is found, not an accident of how many *quoted*
        data-state attributes happen to exist on a given page.

        Inputs:
            html_text: raw HTML of a guide detail page.
        Output: the parsed "guide" dict, verbatim/unreshaped — this
            container's boundary (see module docstring) is to unwrap
            the container, not reshape its contents, so no fields are
            selected or dropped here.
        Side effects: none — pure parsing, no network or disk I/O.
        Exceptions: raises ValueError if div.wrapper > div.content >
            div#root can't be found, if it has no data-state attribute,
            if the attribute's content isn't valid HTML-escaped JSON,
            or if the parsed JSON has no "guide" key — a genuine
            site-format change should be loud, not silently produce an
            empty/wrong file.
        """
        soup = BeautifulSoup(html_text, "html.parser")
        root_div = soup.select_one(_ROOT_DIV_SELECTOR)
        if root_div is None:
            raise ValueError(
                f"No element matching {_ROOT_DIV_SELECTOR!r} found in guide detail page"
            )

        data_state = root_div.get("data-state")
        if not isinstance(data_state, str):
            raise ValueError(
                f"{_ROOT_DIV_SELECTOR!r} element has no data-state attribute"
            )

        # BeautifulSoup already decodes character references in
        # attribute values while parsing, so this is normally a no-op
        # — kept as cheap defense-in-depth in case that ever isn't
        # true for some entity form, rather than a load-bearing step.
        raw_json = html.unescape(data_state)
        payload = json.loads(raw_json)

        if "guide" not in payload:
            raise ValueError(
                f"data-state payload has no 'guide' key: {sorted(payload.keys())}"
            )

        return payload["guide"]
