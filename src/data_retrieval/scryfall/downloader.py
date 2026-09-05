"""Downloads and extracts the Scryfall oracle-cards data dump.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).
"""

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

from src.data_retrieval.download_utils import download_to_file


@dataclass
class ScryfallDownloadResult:
    """Paths to the two files a completed fetch leaves on disk.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    compressed_path: Path  # the downloaded .jsonl.gz, as-is from Scryfall
    jsonl_path: Path  # the extracted .jsonl, sibling of compressed_path


class ScryfallOracleDownloader:
    """Downloads and extracts one Scryfall oracle-cards bulk data file.

    Single-consumer to src/data_retrieval/scryfall/ — no other source
    directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/scryfall")

    def __init__(self, source_url: str, raw_data_dir: Path | None = None) -> None:
        """
        Inputs:
            source_url: full URL to a Scryfall oracle-cards
                .jsonl.gz bulk data file (e.g.
                "https://data.scryfall.io/oracle-cards/oracle-cards-<ts>.jsonl.gz").
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
        self.source_url = source_url
        self.raw_data_dir = raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR

    def fetch(self) -> ScryfallDownloadResult:
        """Entry point: download the bulk file, then extract it.

        Composed of download() followed by extract() — no additional
        logic beyond calling the two and returning their result.

        Inputs: none (uses self.source_url, self.raw_data_dir).
        Output: ScryfallDownloadResult with both resulting paths.
        Side effects: one network request; writes two files to
            raw_data_dir.
        Exceptions: whatever download() or extract() raise.

        Example:
            >>> downloader = ScryfallOracleDownloader(
            ...     "https://data.scryfall.io/oracle-cards/oracle-cards-20260820090157.jsonl.gz",
            ... )
            >>> result = downloader.fetch()
        """
        compressed_path = self.download()
        jsonl_path = self.extract(compressed_path)
        return ScryfallDownloadResult(
            compressed_path=compressed_path, jsonl_path=jsonl_path
        )

    def download(self) -> Path:
        """Download self.source_url into self.raw_data_dir, unmodified.

        Inputs: none (uses self.source_url, self.raw_data_dir).
        Output: path to the saved .jsonl.gz file.
        Side effects: one network request; creates raw_data_dir if
            missing; writes one file to disk.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file. Any
            partially-written file is removed before the exception
            propagates, so a failed download never leaves a truncated
            file behind in raw_data_dir.
        """
        filename = Path(urlparse(self.source_url).path).name
        destination_path = self.raw_data_dir / filename

        download_to_file(self.source_url, destination_path)

        return destination_path

    def extract(self, compressed_path: Path) -> Path:
        """Decompress a .jsonl.gz file to a sibling .jsonl file.

        Inputs:
            compressed_path: path to a .jsonl.gz file (e.g. one
                returned by download()).
        Output: path to the extracted .jsonl file, in the same
            directory as compressed_path.
        Side effects: writes one file to disk.
        Exceptions: raises if compressed_path doesn't exist, isn't a
            valid gzip file, or the extracted file can't be written.
            Any partially-written file is removed before the exception
            propagates.
        """
        jsonl_path = compressed_path.with_suffix("")

        try:
            with gzip.open(compressed_path, "rb") as compressed_file:
                with open(jsonl_path, "wb") as jsonl_file:
                    shutil.copyfileobj(compressed_file, jsonl_file)
        except Exception:
            jsonl_path.unlink(missing_ok=True)
            raise

        return jsonl_path
