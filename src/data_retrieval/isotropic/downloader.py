"""Downloads whatever isotropic.org Dominion game-log files survive in
the Wayback Machine — the origin server's own download links now 404,
Wayback is all that's left.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing (that's data_refinement's job). See
src/data_retrieval/dominion/todo.md for how this source was found and
why it matters (real bulk historical Dominion game data, the one
credible lead of its kind — see that file's "Real played decks / game
logs" section).

isotropic.org (an early, now-defunct fan-run Dominion server) used to
publish per-day full game logs and per-month one-line-per-game summary
logs under /gamelog/, but historically changed its own URL shape at
least once — e.g. both
`https://dominion.isotropic.org/gamelog/2010/20101011.tar.bz2` and the
older `http://dominion.isotropic.org/gamelog/201010/11/all.tar.bz2`
archive the same October 11, 2010 log. Because of that, this class
does NOT guess at a fixed URL template and probe it — an earlier
version of this file did exactly that (one CDX lookup per candidate
month, assuming the modern `{year}/{year}{month:02d}-summary.tar.bz2`
shape), and it was both slow (~28 separate paced requests, several of
which hit real archive.org 503s under live testing) and blind to any
URL shape it didn't already know to guess.

Instead, phase_1() makes exactly ONE call to the Wayback CDX API with
`matchType=prefix` against the whole `dominion.isotropic.org/gamelog/`
path, which returns every URL under that prefix Wayback has a 200-status
snapshot of, in whatever shape it's actually in. phase_2() downloads
each one.

CONFIRMED LIVE (2026-09-17, via this exact prefix query): only 5 URLs
exist in the whole archive, total — matching the human's own manual
finding of "the 4 links on the main isotropic.org page" plus one
legacy-URL-shape duplicate of one of them:
    https://dominion.isotropic.org/gamelog/2010/20101011.tar.bz2
    https://dominion.isotropic.org/gamelog/2010/201012-summary.tar.bz2
    http://dominion.isotropic.org/gamelog/201010/11/all.tar.bz2   (dup of the first, old URL shape)
    https://dominion.isotropic.org/gamelog/2013/201303-summary.tar.bz2
    https://dominion.isotropic.org/gamelog/2013/20130315.tar.bz2
The vast majority of the real historical corpus (~13GB of daily logs,
~1GB of monthly summaries, per dominion/todo.md's earlier estimate)
was apparently never crawled by Wayback at all, and is gone.

Downloads use Wayback's `id_` URL modifier
(`/web/{timestamp}id_/{original_url}`), which serves the archived
resource's raw bytes exactly as captured, without Wayback's usual
banner/link-rewriting injection — the right modifier for a binary
download like a .tar.bz2, not the plain `/web/{timestamp}/{url}` form
meant for browsing archived HTML pages.
"""

import json
from pathlib import Path
from typing import ClassVar, NamedTuple

from tqdm import tqdm

from src.data_retrieval.download_utils import download_to_file, download_to_string
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter


class _ArchivedFile(NamedTuple):
    """One Wayback-archived isotropic.org gamelog file, as recorded in
    archived_files.jsonl by phase_1() and read back by phase_2(). Not
    part of this module's public surface — phase_2() only ever gets
    these by reading them back off disk, never directly from phase_1()
    in memory, matching every other id-list-then-detail source in this
    container (e.g. STSGGRunDownloader)."""

    original_url: str
    timestamp: str


class IsotropicGameLogDownloader(Downloader):
    """Downloads whatever Wayback-archived isotropic.org gamelog files
    still exist, whatever their URL shape or content (daily full logs,
    monthly summaries, or any other file that ever lived under
    /gamelog/) — see this module's docstring for why this doesn't
    filter by filename pattern.

    Single-consumer to src/data_retrieval/isotropic/ — no other source
    directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/isotropic")

    GAMELOG_PREFIX: ClassVar[str] = "dominion.isotropic.org/gamelog/"
    CDX_PREFIX_API_URL_TEMPLATE: ClassVar[str] = (
        "https://web.archive.org/cdx/search/cdx"
        "?url={prefix}&matchType=prefix&output=json&filter=statuscode:200"
    )
    WAYBACK_DOWNLOAD_URL_TEMPLATE: ClassVar[str] = (
        "https://web.archive.org/web/{timestamp}id_/{original_url}"
    )

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__. Shared across both
                the CDX lookup and every Wayback file download — both
                go to archive.org, so one shared pace is the right
                politeness unit here, not two independent ones.
            raw_data_dir: see Downloader.__init__.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/phase_2()
            are called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)

    def _run_phase_1(self) -> Path:
        """Make one prefix-match CDX API call against GAMELOG_PREFIX
        and record every 200-status URL it returns, deduplicated by
        original_url (keeping the earliest timestamp on a duplicate —
        an arbitrary but stable tie-break; this file's whole premise is
        "does anything survive at all," so among several snapshots of
        the same URL there's no other principled way to rank them).

        Inputs: none (uses GAMELOG_PREFIX, CDX_PREFIX_API_URL_TEMPLATE).
        Output: path to the written file list
            (raw_data_dir/archived_files.jsonl), one _ArchivedFile per
            line.
        Side effects: one paced network request; creates raw_data_dir
            if missing; writes raw_data_dir/archived_files.jsonl,
            overwriting any existing one.
        Exceptions: raises on a CDX API call that still fails after
            download_to_string()'s retries, or if the response body
            isn't valid JSON — a single request with nothing else to
            fall back to, unlike phase_2()'s per-file best-effort
            handling.

        Example:
            >>> downloader = IsotropicGameLogDownloader()
            >>> files_path = downloader.phase_1()
        """
        self.rate_limiter.wait()
        url = self.CDX_PREFIX_API_URL_TEMPLATE.format(prefix=self.GAMELOG_PREFIX)
        rows = json.loads(download_to_string(url))

        earliest_timestamp_by_url: dict[str, str] = {}
        if len(rows) > 1:
            # First row is always the header; no further rows means no
            # matching file anywhere under GAMELOG_PREFIX.
            header, *file_rows = rows
            url_index = header.index("original")
            timestamp_index = header.index("timestamp")
            for row in file_rows:
                original_url = row[url_index]
                timestamp = row[timestamp_index]
                existing = earliest_timestamp_by_url.get(original_url)
                if existing is None or timestamp < existing:
                    earliest_timestamp_by_url[original_url] = timestamp

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        files_path = self.raw_data_dir / "archived_files.jsonl"
        files_path.write_text(
            "".join(
                json.dumps(_ArchivedFile(url, timestamp)._asdict()) + "\n"
                for url, timestamp in earliest_timestamp_by_url.items()
            ),
            encoding="utf-8",
        )

        return files_path

    def phase_2(self) -> Path:
        """Download every file recorded by phase_1() into
        raw_data_dir, skipping any already recorded in
        downloads_manifest.txt.

        Inputs: none (uses self.raw_data_dir, self.rate_limiter —
            reads raw_data_dir/archived_files.jsonl, written by
            phase_1()).
        Output: raw_data_dir itself — the directory every downloaded
            file lands in (there's no single combined output file;
            each archived file is its own tarball).
        Side effects: one paced network request per file not already
            in downloads_manifest.txt; writes one file per successful
            download, named after original_url's path (see
            _destination_filename()); appends one line to
            downloads_manifest.txt per successful download, flushing
            immediately; prints a tqdm progress bar to stderr; prints
            one tqdm.write() line per file that fails and gets skipped.
        Exceptions: raises only if raw_data_dir/archived_files.jsonl
            doesn't exist (phase_1() hasn't been run) — per-file
            download failures are caught internally and skipped, never
            propagated (best-effort, same convention as
            STSGGRunDownloader.phase_2: one unhealthy file — there are
            only a handful total — shouldn't lose the rest).

        Example:
            >>> downloader = IsotropicGameLogDownloader()
            >>> downloader.phase_1()
            >>> downloader.phase_2()
        """
        files_path = self.raw_data_dir / "archived_files.jsonl"
        archived_files = [
            _ArchivedFile(**json.loads(line))
            for line in files_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        manifest_path = self.raw_data_dir / "downloads_manifest.txt"
        already_downloaded = (
            {
                line
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if manifest_path.exists()
            else set()
        )

        for archived_file in tqdm(
            archived_files, desc="isotropic downloads", unit="file"
        ):
            if archived_file.original_url in already_downloaded:
                continue

            destination_path = self.raw_data_dir / self._destination_filename(
                archived_file.original_url
            )
            download_url = self.WAYBACK_DOWNLOAD_URL_TEMPLATE.format(
                timestamp=archived_file.timestamp,
                original_url=archived_file.original_url,
            )

            self.rate_limiter.wait()
            try:
                download_to_file(download_url, destination_path)
            except Exception as error:
                # Best-effort — see this method's docstring.
                tqdm.write(
                    f"isotropic download failed ({archived_file.original_url}): {error}"
                )
                continue

            with open(manifest_path, "a", encoding="utf-8") as manifest_file:
                manifest_file.write(archived_file.original_url + "\n")
                manifest_file.flush()

        return self.raw_data_dir

    def _destination_filename(self, original_url: str) -> str:
        """Derive a filesystem-safe, collision-resistant filename for
        original_url's downloaded content.

        isotropic.org used more than one URL shape over its lifetime
        (see this module's docstring) — some archived paths are just
        `{year}/{file}`, others nest further (`{year}{month}/{day}/
        all.tar.bz2`). Rather than assume any one shape, this takes
        every path segment after GAMELOG_PREFIX and joins them
        with "_", which is stable across whatever shape a given URL
        turns out to have and avoids two different archived paths
        colliding on the same output filename.

        Inputs:
            original_url: an isotropic.org URL starting with
                "https://" or "http://" + GAMELOG_PREFIX.
        Output: a filename (no directory component), e.g.
            "2010_20101011.tar.bz2" for
            ".../gamelog/2010/20101011.tar.bz2", or
            "201010_11_all.tar.bz2" for
            ".../gamelog/201010/11/all.tar.bz2".
        Side effects: none.
        Exceptions: none expected for a URL actually containing
            GAMELOG_PREFIX (true of everything phase_1() records).

        Example:
            >>> IsotropicGameLogDownloader()._destination_filename(
            ...     "https://dominion.isotropic.org/gamelog/2010/20101011.tar.bz2"
            ... )
            '2010_20101011.tar.bz2'
        """
        _, _, path_after_prefix = original_url.partition(self.GAMELOG_PREFIX)
        return path_after_prefix.replace("/", "_")
