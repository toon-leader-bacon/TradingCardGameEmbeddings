"""Downloads sts2runs.com's monthly Slay the Spire 2 run-data dump.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

sts2runs.com publishes a monthly snapshot of community-submitted runs
as a single gzip-compressed NDJSON file (one run object per line, each
tagged with _serverId/_isCheated/_cheatedReasons) under a dated
filename at https://sts2runs.com/downloads/ — no API, no pagination,
no auth, but the filename changes with every monthly snapshot (like
Scryfall's oracle-cards dumps). DEFAULT_SOURCE_URL is only ever the
snapshot current as of when this was written — unlike
SpireCodexCardDownloader's genuinely fixed URL, a caller after a new
monthly snapshot has been published should pass the current one in
explicitly rather than relying on the default going stale.

download()/extract() below are structurally identical to
ScryfallOracleDownloader's (src/data_retrieval/scryfall/downloader.py)
— same dated-URL-in/download-then-gzip-extract shape. Left duplicated
rather than extracted into download_utils.py: this is only the second
occurrence of that shape (Scryfall being the first), and this
container's own download_utils.py deliberately waited for a third
occurrence ("rule of three") before extracting download_to_file/
download_to_string — see src/data_retrieval/README.md. If a third
dated-gzip-dump source shows up, extract a shared
download_and_extract_gzip() helper into download_utils.py then, rather
than letting a fourth copy accumulate.
"""

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

from src.data_retrieval.download_utils import download_to_file


@dataclass
class STS2RunsDownloadResult:
    """Paths to the two files a completed fetch leaves on disk.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    compressed_path: Path  # the downloaded .json.gz, as-is from sts2runs.com
    ndjson_path: Path  # the extracted NDJSON file, sibling of compressed_path


class STS2RunsDownloader:
    """Downloads and extracts one sts2runs.com monthly run-data snapshot.

    Single-consumer to src/data_retrieval/sts2runs/ — no other source
    directory depends on this class. Mirrors ScryfallOracleDownloader's
    shape (dynamic dated URL, download-then-gzip-extract) rather than
    SpireCodexCardDownloader's (fixed URL, no extraction) since both
    the URL and the compression match Scryfall's case, not
    spire-codex's.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/sts2runs")
    DEFAULT_SOURCE_URL: ClassVar[str] = (
        "https://sts2runs.com/downloads/runs-all-before-2026-06.json.gz"
    )

    def __init__(
        self, source_url: str | None = None, raw_data_dir: Path | None = None
    ) -> None:
        """
        Inputs:
            source_url: full URL to a specific sts2runs.com monthly
                dump (e.g.
                "https://sts2runs.com/downloads/runs-all-before-2026-06.json.gz").
                Defaults to DEFAULT_SOURCE_URL when omitted, but that
                constant only reflects the snapshot current as of when
                this class was written — check
                https://sts2runs.com/downloads for a newer one and pass
                it explicitly once it's stale.
            raw_data_dir: directory both the compressed and extracted
                files are written into. Defaults to
                DEFAULT_RAW_DATA_DIR when omitted (expected to be a
                path under data/raw, per src/README.md — not this
                class's concern to enforce, just to receive).
        Output: none (constructor).
        Side effects: none — no I/O happens until fetch()/download()/
            extract() are called.
        Exceptions: none.
        """
        self.source_url = source_url if source_url is not None else self.DEFAULT_SOURCE_URL
        self.raw_data_dir = (
            raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR
        )

    def fetch(self) -> STS2RunsDownloadResult:
        """Entry point: download the monthly snapshot, then extract it.

        Composed of download() followed by extract() — no additional
        logic beyond calling the two and returning their result.

        Inputs: none (uses self.source_url, self.raw_data_dir).
        Output: STS2RunsDownloadResult with both resulting paths.
        Side effects: one network request; writes two files to
            raw_data_dir.
        Exceptions: whatever download() or extract() raise.

        Example:
            >>> downloader = STS2RunsDownloader(
            ...     "https://sts2runs.com/downloads/runs-all-before-2026-06.json.gz",
            ... )
            >>> result = downloader.fetch()
        """
        compressed_path = self.download()
        ndjson_path = self.extract(compressed_path)
        return STS2RunsDownloadResult(
            compressed_path=compressed_path, ndjson_path=ndjson_path
        )

    def download(self) -> Path:
        """Download self.source_url into self.raw_data_dir, unmodified.

        Inputs: none (uses self.source_url, self.raw_data_dir).
        Output: path to the saved .json.gz file.
        Side effects: one network request; creates raw_data_dir if
            missing; writes one file to disk.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file. Any
            partially-written file is removed before the exception
            propagates (download_to_file's behavior).
        """
        filename = Path(urlparse(self.source_url).path).name
        destination_path = self.raw_data_dir / filename

        download_to_file(self.source_url, destination_path)

        return destination_path

    def extract(self, compressed_path: Path) -> Path:
        """Decompress a .json.gz snapshot to a sibling NDJSON file.

        The result is still one JSON object per line (NDJSON), despite
        the source naming the compressed file .json.gz rather than
        .ndjson.gz — this only strips the .gz suffix, it doesn't rename
        the inner format.

        Inputs:
            compressed_path: path to a .json.gz file (e.g. one
                returned by download()).
        Output: path to the extracted NDJSON file, in the same
            directory as compressed_path.
        Side effects: writes one file to disk.
        Exceptions: raises if compressed_path doesn't exist, isn't a
            valid gzip file, or the extracted file can't be written.
            Any partially-written file is removed before the exception
            propagates.
        """
        ndjson_path = compressed_path.with_suffix("")

        try:
            with gzip.open(compressed_path, "rb") as compressed_file:
                with open(ndjson_path, "wb") as ndjson_file:
                    shutil.copyfileobj(compressed_file, ndjson_file)
        except Exception:
            ndjson_path.unlink(missing_ok=True)
            raise

        return ndjson_path
