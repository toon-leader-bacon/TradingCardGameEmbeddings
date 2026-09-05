"""Downloads spire-codex's card data dump from its GitHub repo.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

spire-codex (github.com/ptrlrd/spire-codex) publishes its full card
list as a single static JSON file committed to the repo itself
(data/eng/cards.json), fetched via GitHub's raw-content host — no API,
no pagination, no auth. One request gets everything, so unlike
HearthstoneJsonDownloader or GwentOneDownloader there's no listing
phase and no rate limiter dependency.
"""

from pathlib import Path
from typing import ClassVar

from src.data_retrieval.download_utils import download_to_file


class SpireCodexCardDownloader:
    """Downloads spire-codex's full cards.json in a single request.

    Single-consumer to src/data_retrieval/spire_codex/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/spire_codex")
    DEFAULT_CARDS_URL: ClassVar[str] = (
        "https://raw.githubusercontent.com/ptrlrd/spire-codex/main/data/eng/cards.json"
    )

    def __init__(
        self, cards_url: str | None = None, raw_data_dir: Path | None = None
    ) -> None:
        """
        Inputs:
            cards_url: URL of the cards.json file to download. Defaults
                to DEFAULT_CARDS_URL when omitted.
            raw_data_dir: directory cards.json is written into.
                Defaults to DEFAULT_RAW_DATA_DIR when omitted (expected
                to be a path under data/raw, per src/README.md — not
                this class's concern to enforce, just to receive).
        Output: none (constructor).
        Side effects: none — no I/O happens until fetch() is called.
        Exceptions: none.
        """
        self.cards_url = cards_url if cards_url is not None else self.DEFAULT_CARDS_URL
        self.raw_data_dir = (
            raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR
        )

    def fetch(self) -> Path:
        """Download cards.json and save it to raw_data_dir, always
        overwriting any existing file.

        Always-overwrite (no skip-if-exists) is a deliberate departure
        from HearthstoneJsonDownloader's resumability: that source's
        per-build files are immutable history, whereas spire-codex's
        single cards.json reflects the *current* card pool — a stale
        on-disk copy would silently go unrefreshed if this skipped an
        existing file the way that downloader does.

        Inputs: none (uses self.cards_url, self.raw_data_dir).
        Output: path to the saved cards.json
            (raw_data_dir/cards.json).
        Side effects: one network request; creates raw_data_dir if
            missing; writes one file. Any partially-written file is
            removed before the exception propagates
            (download_to_file's behavior).
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file.

        Example:
            >>> downloader = SpireCodexCardDownloader()
            >>> path = downloader.fetch()
        """
        destination_path = self.raw_data_dir / "cards.json"
        download_to_file(self.cards_url, destination_path)
        return destination_path
