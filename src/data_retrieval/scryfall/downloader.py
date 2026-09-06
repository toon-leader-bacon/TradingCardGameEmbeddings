"""Downloads and extracts the Scryfall oracle-cards data dump.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).
"""

import gzip
import shutil
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter


class ScryfallOracleDownloader(Downloader):
    """Downloads and extracts one Scryfall oracle-cards bulk data file.

    Single-consumer to src/data_retrieval/scryfall/ — no other source
    directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/scryfall")

    def __init__(
        self,
        source_url: str,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
    ) -> None:
        """
        Inputs:
            source_url: full URL to a Scryfall oracle-cards
                .jsonl.gz bulk data file (e.g.
                "https://data.scryfall.io/oracle-cards/oracle-cards-<ts>.jsonl.gz").
                Required and positional (unlike rate_limiter/
                raw_data_dir): there is no sensible default, since
                Scryfall's dump URLs are dated and change with every
                bulk-data release.
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/download()/
            extract() are called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.source_url = source_url

    def _run_phase_1(self) -> Path:
        """Download the bulk file, then extract it, returning the
        extracted file's path — the artifact any downstream consumer
        actually reads.

        Composed of download() followed by extract() — no additional
        logic beyond calling the two and returning extract()'s result.
        The intermediate compressed file stays reachable via download()
        directly, for a caller that specifically wants it.

        Inputs: none (uses self.source_url, self.raw_data_dir).
        Output: path to the extracted .jsonl file.
        Side effects: one network request; writes two files to
            raw_data_dir.
        Exceptions: whatever download() or extract() raise.

        Example:
            >>> downloader = ScryfallOracleDownloader(
            ...     "https://data.scryfall.io/oracle-cards/oracle-cards-20260820090157.jsonl.gz",
            ... )
            >>> jsonl_path = downloader.phase_1()
        """
        compressed_path = self.download()
        return self.extract(compressed_path)

    def download(self) -> Path:
        """Download self.source_url into self.raw_data_dir, unmodified.

        Inputs: none (uses self.source_url, self.raw_data_dir,
            self.rate_limiter).
        Output: path to the saved .jsonl.gz file.
        Side effects: one paced network request; creates raw_data_dir
            if missing; writes one file to disk.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file. Any
            partially-written file is removed before the exception
            propagates, so a failed download never leaves a truncated
            file behind in raw_data_dir.
        """
        filename = Path(urlparse(self.source_url).path).name
        destination_path = self.raw_data_dir / filename

        self.rate_limiter.wait()
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
