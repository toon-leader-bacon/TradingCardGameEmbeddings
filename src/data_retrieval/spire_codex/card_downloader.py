"""Downloads spire-codex's card data dump from its GitHub repo.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

spire-codex (github.com/ptrlrd/spire-codex) publishes its full card
list as a single static JSON file committed to the repo itself
(data/eng/cards.json), fetched via GitHub's raw-content host — no API,
no pagination, no auth. One request gets everything, so unlike
HearthstoneJsonDownloader or GwentOneDownloader there's no listing
phase — but it still takes and uses a RateLimiter, like every source
in this container (see Downloader), even though it only ever makes one
request.
"""

from pathlib import Path
from typing import ClassVar

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter


class SpireCodexCardDownloader(Downloader):
    """Downloads spire-codex's full cards.json in a single request.

    Single-consumer to src/data_retrieval/spire_codex/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/spire_codex")
    DEFAULT_CARDS_URL: ClassVar[str] = (
        "https://raw.githubusercontent.com/ptrlrd/spire-codex/main/data/eng/cards.json"
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
            cards_url: URL of the cards.json file to download. Defaults
                to DEFAULT_CARDS_URL when omitted.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1() is called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.cards_url = cards_url if cards_url is not None else self.DEFAULT_CARDS_URL

    def _run_phase_1(self) -> Path:
        """Download cards.json and save it to raw_data_dir, always
        overwriting any existing file.

        Always-overwrite (no skip-if-exists) is a deliberate departure
        from HearthstoneJsonDownloader's resumability: that source's
        per-build files are immutable history, whereas spire-codex's
        single cards.json reflects the *current* card pool — a stale
        on-disk copy would silently go unrefreshed if this skipped an
        existing file the way that downloader does.

        Inputs: none (uses self.cards_url, self.raw_data_dir,
            self.rate_limiter).
        Output: path to the saved cards.json
            (raw_data_dir/cards.json).
        Side effects: one paced network request; creates raw_data_dir
            if missing; writes one file. Any partially-written file is
            removed before the exception propagates
            (download_to_file's behavior).
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file.
        """
        destination_path = self.raw_data_dir / "cards.json"
        self.rate_limiter.wait()
        download_to_file(self.cards_url, destination_path)
        return destination_path
