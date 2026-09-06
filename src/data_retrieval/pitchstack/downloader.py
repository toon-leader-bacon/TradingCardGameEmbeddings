"""Downloads Flesh and Blood deck data from pitchstack.gg, in two
phases.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job) — whatever this project's own card identity
scheme (nocab_uuid) turns out to need from pitchstack's own
cardId/deck ids is explicitly out of scope here, deferred to a future
data_refinement pass. See plans/pitchstack.md for this feature's scope
and open questions.

Two phases, kept as two separate public methods rather than composed
into one fetch() — same rationale as PlayGwentDownloader/
STSGGRunDownloader: they're expected to be run and re-run as separate
steps against a live site. phase_2()'s resumability (manifest check,
data-row-before-manifest-row append) is shared with those two sibling
downloaders via download_utils.read_manifest()/append_with_manifest()
rather than reimplemented here — see that module's docstring.

    1. phase_1() — fetches pitchstack.gg's public-decks sitemap
       (a standard XML sitemap, GET
       https://pitchstack.gg/sitemaps/public-decks-0.xml) and extracts
       every deck id ("d-<uuid>", e.g.
       "d-f947a7e9-699b-5938-9db0-2edca16c6c40") found in its <loc>
       URLs, writing them to deck_ids.txt.

       CONFIRMED LIVE (2026-09-06, curl against the real sitemap): the
       id embedded in each <loc> is a plain deck id, e.g.
       https://pitchstack.gg/decks/d-f947a7e9-699b-5938-9db0-2edca16c6c40
       — NOT a deck *version* id ("dv-<uuid>"), despite an earlier
       version of this module assuming the sitemap's id was already a
       deck version id extractable straight into
       GET /v1/deck_versions/{deck_version_id}/cards. It isn't: a deck
       id and its deck version id(s) are unrelated UUIDs (not one
       derived from the other), and the only confirmed way to get from
       one to the other is phase_2()'s own /v1/decks/{deck_id}
       response, which embeds "activeDeckVersionId" and a full
       "deckVersions" list. See plans/pitchstack.md.

    2. phase_2() — reads deck_ids.txt and, for every id not already
       recorded in decks_manifest.txt, fetches that deck's full record
       (GET https://api.pitchstack.gg/v1/decks/{deck_id} — CONFIRMED
       LIVE, 2026-09-06; note this has NO "/meta" suffix, despite an
       earlier guess in this module) and appends the response JSON,
       verbatim, to decks.jsonl, via the shared read_manifest()/
       append_with_manifest() helpers in download_utils.py (see that
       module's docstring — extracted from this exact shape once it
       showed up independently in STSGGRunDownloader.phase_2 and
       PlayGwentDownloader.phase_2 too). Those helpers guarantee the
       data row is written before the manifest row, so a crash between
       the two leaves a harmless, re-fetchable gap (a duplicate line a
       downstream reader can de-dup on) rather than a manifest falsely
       claiming a deck is done while its data row was never written.

       A confirmed-live sample response:
       {"deck": {"id": "d-...", "userId": ..., "name": ..., "author":
       ..., "heroId": ..., "format": ..., "visibility": ...,
       "createdAt": ..., "updatedAt": ..., "deckVersions": [{"id":
       "dv-...", "name": ...}, ...], "activeDeckVersionId": "dv-...",
       "deckKind": ..., "sourceKind": ..., "sourceReference": ...,
       "tournamentType": ..., "eventId": ...}}. This class stores
       whatever the real payload contains as-is, unreshaped, per this
       container's fetch-only contract — it does not assume any of the
       above are the payload's only keys.

Two more pitchstack.gg endpoints exist and are deliberately NOT
implemented by this class yet (see plans/pitchstack.md):

    - GET /v1/deck_versions/{deck_version_id}/cards — one deck
      version's card list. phase_2()'s /v1/decks/{deck_id} response
      already gives every deck_version_id needed to call this
      (activeDeckVersionId, or each entry in deckVersions) — collecting
      card lists is future work building on top of decks.jsonl, not
      this class's job right now.
    - GET /v1/deck_versions/{deck_version_id}/history — a deck
      version's edit history (cards added/removed between versions).
"""

import json
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import ClassVar

from tqdm import tqdm

from src.data_retrieval.download_utils import (
    append_with_manifest,
    download_to_string,
    read_manifest,
)
from src.data_retrieval.rate_limiter import RateLimiter

# Sitemaps use this namespace on every element, including <loc> — see
# https://www.sitemaps.org/schemas/sitemap/0.9 — ElementTree requires
# it spelled out on every tag lookup, it isn't inferred from the
# document's own xmlns declaration the way some other XML APIs do.
_SITEMAP_XML_NAMESPACE = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
# Precise UUID group structure (rather than a loose "36 hex-or-hyphen
# characters" catchall) plus lookaround asserting "d-" isn't itself
# part of a longer token (e.g. mid-word, or the "dv-" prefix a deck
# *version* id uses) — <loc> is a full URL, and this is deliberately
# tighter than "search anywhere in the string" would need, so a future
# sitemap change that adds an unrelated "...id-<hex>..." token
# elsewhere in the URL can't be silently mismatched for a deck id.
_DECK_ID_PATTERN = re.compile(
    r"(?<![\w-])d-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![\w-])"
)


class PitchstackDeckDownloader:
    """Downloads Flesh and Blood deck ids and per-deck records from
    pitchstack.gg.

    Single-consumer to src/data_retrieval/pitchstack/ — no other source
    directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/pitchstack")
    DEFAULT_SITEMAP_URL: ClassVar[str] = (
        "https://pitchstack.gg/sitemaps/public-decks-0.xml"
    )
    DEFAULT_DECK_URL: ClassVar[str] = "https://api.pitchstack.gg/v1/decks/{deck_id}"

    def __init__(
        self,
        rate_limiter: RateLimiter,
        output_dir: Path | None = None,
        sitemap_url: str | None = None,
        deck_url: str | None = None,
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
                (deck_ids.txt, decks.jsonl, decks_manifest.txt).
                Defaults to DEFAULT_RAW_DATA_DIR when omitted (expected
                to be a path under data/raw, per src/README.md — not
                this class's concern to enforce, just to receive).
            sitemap_url: URL of the public-decks sitemap. Defaults to
                DEFAULT_SITEMAP_URL when omitted.
            deck_url: URL template for one deck's record JSON,
                containing a "{deck_id}" placeholder. Defaults to
                DEFAULT_DECK_URL when omitted.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/phase_2()
            are called.
        Exceptions: none.
        """
        self.rate_limiter = rate_limiter
        self.output_dir = output_dir or self.DEFAULT_RAW_DATA_DIR
        self.sitemap_url = sitemap_url or self.DEFAULT_SITEMAP_URL
        self.deck_url = deck_url or self.DEFAULT_DECK_URL

    def phase_1(self) -> Path:
        """Fetch the public-decks sitemap and write every deck id found
        in it to disk.

        Inputs: none (uses self.sitemap_url, self.output_dir,
            self.rate_limiter).
        Output: path to the written ids file (output_dir/deck_ids.txt).
        Side effects: one paced network request (via
            download_to_string(), which retries internally); creates
            output_dir if missing; writes one file
            (output_dir/deck_ids.txt), overwriting any existing one.
        Exceptions: raises on a request that still fails after
            download_to_string()'s retries, if the response body isn't
            valid XML, or if it contains no <loc> entries with an
            extractable deck id (a genuine site-format change should be
            loud, not silently produce an empty file).

        Example:
            >>> downloader = PitchstackDeckDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> ids_path = downloader.phase_1()
        """
        result: Path

        self.rate_limiter.wait()
        sitemap_xml = download_to_string(self.sitemap_url)

        # Parse every <loc> URL out of the sitemap, then extract this
        # source's deck id from each one.
        deck_ids = self._extract_deck_ids(sitemap_xml)

        if not deck_ids:
            raise ValueError(f"No deck ids found in sitemap at {self.sitemap_url!r}")

        # Dedup preserving first-seen order, same rationale as
        # PlayGwentDownloader.phase_1/STSGGRunDownloader.phase_1: a
        # duplicate seen mid-parse could be a real site signal, not
        # hidden by deduplicating any earlier than this.
        deduplicated_ids = list(dict.fromkeys(deck_ids))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        result = self.output_dir / "deck_ids.txt"
        result.write_text(
            "".join(f"{deck_id}\n" for deck_id in deduplicated_ids),
            encoding="utf-8",
        )

        return result

    def phase_2(self) -> Path:
        """For every deck id not already recorded in decks_manifest.txt,
        fetch that deck's full record and append it to decks.jsonl.

        decks.jsonl is the actual data (one JSON object per line,
        appended, never rewritten); decks_manifest.txt is purely a fast
        resumability index over it, not a second source of truth. Same
        data-row-before-manifest-row crash-safety convention as
        STSGGRunDownloader.phase_2 — see this module's docstring.

        A deck that still fails after download_to_string()'s retries is
        logged via tqdm.write() and skipped, not raised — best-effort,
        same convention as the sibling downloaders: a run over many
        thousands of decks shouldn't be lost to one bad id.

        Uses download_utils.read_manifest()/append_with_manifest()
        rather than private helpers of its own — see this module's
        docstring.

        Inputs: none (uses self.output_dir, self.deck_url,
            self.rate_limiter — reads output_dir/deck_ids.txt, written
            by phase_1()).
        Output: path to the JSONL data file (output_dir/decks.jsonl).
        Side effects: one paced network request (more on retry — see
            download_to_string()) per deck id not already in
            decks_manifest.txt; appends one line to decks.jsonl and one
            line to decks_manifest.txt per successfully-fetched deck,
            flushing each write immediately; prints a tqdm progress bar
            to stderr covering all of deck_ids; prints one
            tqdm.write() line per deck that fails and gets skipped.
        Exceptions: raises only if output_dir/deck_ids.txt doesn't
            exist (phase_1() hasn't been run) — per-deck failures are
            caught internally (see above), never propagated.

        Example:
            >>> downloader = PitchstackDeckDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> downloader.phase_1()
            >>> decks_path = downloader.phase_2()
        """
        result: Path

        ids_path = self.output_dir / "deck_ids.txt"
        deck_ids = [
            line
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        manifest_path = self.output_dir / "decks_manifest.txt"
        already_downloaded = read_manifest(manifest_path)

        result = self.output_dir / "decks.jsonl"

        # Fetch each not-yet-downloaded deck, appending its payload
        # (data row) then its id (manifest row) as soon as each one
        # succeeds.
        for deck_id in tqdm(deck_ids, desc="pitchstack.gg decks", unit="deck"):
            if deck_id in already_downloaded:
                continue

            self.rate_limiter.wait()
            try:
                url = self.deck_url.format(deck_id=deck_id)
                deck_payload = json.loads(download_to_string(url))
            except Exception as error:
                tqdm.write(f"pitchstack.gg deck {deck_id} failed: {error}")
                continue

            append_with_manifest(
                result, manifest_path, json.dumps(deck_payload), deck_id
            )

        return result

    def _extract_deck_ids(self, sitemap_xml: str) -> list[str]:
        """Parse a sitemap XML document and return every deck id found
        across its <loc> URLs, in document order.

        Private helper — single consumer is phase_1(). Split out so
        the extraction logic is independently testable against saved
        example sitemap XML without mocking network calls (same
        rationale as PlayGwentDownloader._extract_guide_payload).

        Inputs:
            sitemap_xml: raw XML body of a sitemap document.
        Output: deck ids ("d-<uuid>"), in the order their <loc> entries
            appear in the document. A <loc> URL with no matching id is
            simply skipped, not an error — the sitemap is expected to
            also list non-deck pages.
        Side effects: none — pure parsing, no network or disk I/O.
        Exceptions: raises xml.etree.ElementTree.ParseError if
            sitemap_xml isn't valid XML.
        """
        root = ElementTree.fromstring(sitemap_xml)
        loc_elements = root.findall(".//sitemap:loc", _SITEMAP_XML_NAMESPACE)

        deck_ids = []
        for loc_element in loc_elements:
            if loc_element.text is None:
                continue
            match = _DECK_ID_PATTERN.search(loc_element.text)
            if match is not None:
                deck_ids.append(match.group())

        return deck_ids
