"""Downloads Hearthstone card data dumps from HearthstoneJSON.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

Two phases:
    1. list_build_ids() — fetch the root listing page and parse out
       every build id that has an enUS subdirectory.
    2. download_missing_builds() — for each build id not already
       present in raw_data_dir, download that build's enUS cards.json.
       Best-effort: a build id whose download fails is reported as a
       BuildDownloadFailure rather than aborting the whole run.

Resumability is a plain filesystem check per build id (same pattern as
PokemonTcgDataDownloader) — no separate checkpoint file needed, since
unlike a paginated crawl, build ids don't depend on each other.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import requests
from tqdm import tqdm

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.rate_limiter import RateLimiter

_REQUEST_TIMEOUT_SECONDS = 60
# The live listing page serves relative hrefs (href="/v1/190920/enUS/"),
# not absolute URLs — matching on the path suffix handles that form and
# an absolute-URL form alike, in case that ever changes upstream.
_ENUS_BUILD_LINK_PATTERN = re.compile(r'href="[^"]*/v1/(\d+)/enUS/"')
_CARDS_JSON_URL_TEMPLATE = (
    "https://api.hearthstonejson.com/v1/{build_id}/enUS/cards.json"
)


@dataclass(frozen=True)
class BuildDownloadSuccess:
    """A build id whose cards.json is present in raw_data_dir.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    build_id: str
    path: Path


@dataclass(frozen=True)
class BuildDownloadFailure:
    """A build id whose cards.json could not be downloaded.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    build_id: str
    error: str  # str(exception) — kept as a plain message rather than
    # the exception object itself, so an outcome stays
    # simple, comparable data.


# Discriminated union (PRINCIPLES.md "illegal states unrepresentable" —
# same idiom as GenericCard's tagged union in src/schema/card.py): exactly
# one of "this build succeeded" or "this build failed", never both,
# never neither, and never a partial/ambiguous state in between.
BuildDownloadOutcome = BuildDownloadSuccess | BuildDownloadFailure


class HearthstoneJsonDownloader:
    """Downloads one enUS cards.json per known Hearthstone build.

    Single-consumer to src/data_retrieval/hearthstonejson/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/hearthstonejson")

    def __init__(
        self, listing_url: str, raw_data_dir: Path | None = None, *, rate_limiter: RateLimiter
    ) -> None:
        """
        Inputs:
            listing_url: root HearthstoneJSON listing page URL (e.g.
                "https://api.hearthstonejson.com/v1/").
            raw_data_dir: directory each build's cards.json is written
                into. Defaults to DEFAULT_RAW_DATA_DIR when omitted
                (expected to be a path under data/raw, per
                src/README.md — not this class's concern to enforce,
                just to receive).
            rate_limiter: paces every outgoing request this class
                makes. Passed in rather than constructed internally
                (dependency injection — PATTERNS.md) so the same
                shared RateLimiter instance can be reused across
                sources, and so tests can supply a fast/no-op limiter
                instead of a real one.
        Output: none (constructor).
        Side effects: none — no I/O happens until fetch()/
            list_build_ids()/download_missing_builds()/download_build()
            are called.
        Exceptions: none.
        """
        self.listing_url = listing_url
        self.raw_data_dir = raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR
        self.rate_limiter = rate_limiter

    def fetch(self) -> list[BuildDownloadOutcome]:
        """Entry point: list every known build, then download the
        ones not already present in raw_data_dir.

        Composed of list_build_ids() followed by
        download_missing_builds() — no additional logic beyond calling
        the two and returning the result.

        Inputs: none (uses self.listing_url, self.raw_data_dir,
            self.rate_limiter).
        Output: one BuildDownloadOutcome per known build id —
            BuildDownloadSuccess for a build whose cards.json is
            present in raw_data_dir (already there or newly downloaded
            this run), BuildDownloadFailure for a build whose download
            failed. A single build's failure does not stop the rest —
            see download_missing_builds().
        Side effects: one network request per list_build_ids(), plus
            one more per build actually downloaded; writes one file
            per newly-downloaded, successful build.
        Exceptions: whatever list_build_ids() raises (a failure
            discovering the list of builds is still a hard failure —
            best-effort only applies to individual build downloads,
            see download_missing_builds()).

        Example:
            >>> downloader = HearthstoneJsonDownloader(
            ...     "https://api.hearthstonejson.com/v1/",
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> outcomes = downloader.fetch()
            >>> failures = [o for o in outcomes if isinstance(o, BuildDownloadFailure)]
        """
        build_ids = self.list_build_ids()
        return self.download_missing_builds(build_ids)

    def list_build_ids(self) -> list[str]:
        """Fetch the root listing page and parse out every build id
        that has an enUS subdirectory.

        Composed of one paced GET followed by
        _parse_build_ids_with_enus_subdirectory() — no parsing logic
        lives inline here, so the parsing step can be unit-tested
        against saved example HTML without mocking network calls.

        Inputs: none (uses self.listing_url, self.rate_limiter).
        Output: sorted list of build id strings (e.g. ["190920",
            "191554", ...]). Empty if the listing page's HTML doesn't
            match the expected format — see
            _parse_build_ids_with_enus_subdirectory, which never
            raises on malformed input.
        Side effects: one paced network request.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response). Never raises due to unparseable HTML —
            that case returns an empty list instead (see
            _parse_build_ids_with_enus_subdirectory).
        """
        self.rate_limiter.wait()
        response = requests.get(self.listing_url, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return self._parse_build_ids_with_enus_subdirectory(response.text)

    def _parse_build_ids_with_enus_subdirectory(self, listing_html: str) -> list[str]:
        """Extract every build id that has an enUS subdirectory link.

        Private helper — single consumer is list_build_ids(). Matches
        links ending in "/v1/<build_id>/enUS/" (a `tree`-generated
        HTML index; the live page serves these as relative hrefs, e.g.
        href="/v1/190920/enUS/", not absolute URLs — matching on the
        path suffix handles either form) rather than assuming every
        numeric directory on the listing page has English data.

        Inputs:
            listing_html: raw HTML of the root listing page.
        Output: sorted list of build id strings, deduplicated.
        Side effects: none — pure parsing, no network or disk I/O.
        Exceptions: none expected from well-formed HTML; malformed
            input yields an empty list rather than raising (this is a
            best-effort scrape of a page this project doesn't control).
        """
        build_ids = {
            match.group(1) for match in _ENUS_BUILD_LINK_PATTERN.finditer(listing_html)
        }
        return sorted(build_ids)

    def download_missing_builds(
        self, build_ids: list[str]
    ) -> list[BuildDownloadOutcome]:
        """Download each build id's cards.json that isn't already on disk.

        Best-effort: a build id whose download fails (e.g. an enUS
        listing entry with no cards.json actually behind it — a real
        case seen against the live server) is reported as a
        BuildDownloadFailure and the loop continues to the next build
        id, rather than aborting the whole run. This is a deliberate
        broad `except Exception` — matches the "best effort, one bad
        build shouldn't lose the rest" intent, not an accidental catch-
        all. Every failure is written via tqdm.write() (not print())
        so it doesn't corrupt the progress bar, and is also present in
        the structured return value so callers/tests don't have to
        scrape log output to know what failed.

        Inputs:
            build_ids: build ids to ensure are present in
                raw_data_dir, e.g. ones returned by list_build_ids().
        Output: one BuildDownloadOutcome per input build id, in the
            same order — BuildDownloadSuccess (already present, or
            newly downloaded this call) or BuildDownloadFailure.
        Side effects: creates raw_data_dir if missing; one paced
            network request and (on success) one file write per build
            id not already present; prints a tqdm progress bar to
            stderr covering all of build_ids; prints one tqdm.write()
            line per failed build.
        Exceptions: none — download_build() failures are caught here,
            not propagated. (A failure in list_build_ids(), which this
            method doesn't call, is unaffected by this and still
            raises normally — see fetch().)
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        outcomes: list[BuildDownloadOutcome] = []
        for build_id in tqdm(build_ids, desc="HearthstoneJSON builds", unit="build"):
            destination_path = self.raw_data_dir / f"{build_id}.json"
            if destination_path.exists():
                outcomes.append(
                    BuildDownloadSuccess(build_id=build_id, path=destination_path)
                )
                continue

            try:
                path = self.download_build(build_id)
            except Exception as error:
                tqdm.write(f"HearthstoneJSON build {build_id} failed: {error}")
                outcomes.append(
                    BuildDownloadFailure(build_id=build_id, error=str(error))
                )
            else:
                outcomes.append(BuildDownloadSuccess(build_id=build_id, path=path))

        return outcomes

    def download_build(self, build_id: str) -> Path:
        """Download one build's enUS cards.json.

        Inputs:
            build_id: a Hearthstone build id, e.g. "190920".
        Output: path to the saved cards.json for this build.
        Side effects: one paced network request; creates raw_data_dir
            if missing; writes one file to raw_data_dir. Any
            partially-written file is removed before the exception
            propagates, same convention as the sibling downloaders
            (ScryfallOracleDownloader, PokemonTcgDataDownloader).
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file.
        """
        destination_path = self.raw_data_dir / f"{build_id}.json"
        cards_json_url = _CARDS_JSON_URL_TEMPLATE.format(build_id=build_id)

        self.rate_limiter.wait()
        download_to_file(cards_json_url, destination_path)

        return destination_path
