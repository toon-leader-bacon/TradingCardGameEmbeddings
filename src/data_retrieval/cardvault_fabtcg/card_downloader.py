"""Downloads cardvault.fabtcg.com's public Flesh and Blood card data
dump.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

cardvault.fabtcg.com publishes its full print list as a single static
CSV file served from a CloudFront distribution
(https://d1pn9y8e99aays.cloudfront.net/public_card_data.csv) — no API,
no pagination, no auth. One request gets every print (one row per
card/language/finish combination, ~46,600 rows as of 2026-09-05),
across all print languages the site supports. Structurally mirrors
SpireCodexCardDownloader's single-static-file shape; see that class's
docstring for the same always-overwrite reasoning, which applies here
too — a stale on-disk copy would silently go unrefreshed otherwise.
"""

from pathlib import Path
from typing import ClassVar

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter


class CardVaultFabtcgCardDownloader(Downloader):
    """Downloads cardvault.fabtcg.com's full public card data CSV in a
    single request.

    Single-consumer to src/data_retrieval/cardvault_fabtcg/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/cardvault_fabtcg")
    DEFAULT_CARDS_URL: ClassVar[str] = (
        "https://d1pn9y8e99aays.cloudfront.net/public_card_data.csv"
    )

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        cards_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
            cards_url: URL of the public_card_data.csv file to
                download. Defaults to DEFAULT_CARDS_URL when omitted.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1() is called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.cards_url = cards_url if cards_url is not None else self.DEFAULT_CARDS_URL

    def _run_phase_1(self) -> Path:
        """Download public_card_data.csv and save it to raw_data_dir,
        always overwriting any existing file.

        Inputs: none (uses self.cards_url, self.raw_data_dir,
            self.rate_limiter).
        Output: path to the saved CSV
            (raw_data_dir/public_card_data.csv).
        Side effects: one paced network request; creates raw_data_dir
            if missing; writes one file. Any partially-written file is
            removed before the exception propagates
            (download_to_file's behavior).
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file.
        """
        destination_path = self.raw_data_dir / "public_card_data.csv"
        self.rate_limiter.wait()
        download_to_file(self.cards_url, destination_path)
        return destination_path
