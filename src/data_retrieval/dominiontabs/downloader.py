"""Downloads sumpfork/dominiontabs' Dominion card database — the source
data behind the actively-maintained divider generator at
domdiv.bgtools.net, and the most complete/current card dataset we
found for this game (see src/data_retrieval/dominion/todo.md's "Card
data" section for the comparison against Dominion-app's cards.json and
DominionCardAPI, both dead ends: Dominion-app stops at Nocturne,
missing Menagerie/Allies/Plunder/Rising Sun; DominionCardAPI's hosted
app is offline).

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing (that's data_refinement's job).

dominiontabs splits a card's data across two upstream files rather than
one flat record: `card_db_src/cards_db.json` holds language-neutral
fields (cost, potion/debt cost, types, which expansion(s) a card
belongs to), while a card's actual rules text lives separately per
language under `card_db_src/<locale>/cards_<locale>.json` (e.g.
`en_us/cards_en_us.json`), keyed by card name — a translator can update
one language's text without touching the shared fields. This class
downloads both English-relevant files verbatim; joining them into one
per-card record is a data_refinement concern, not this container's.
"""

from pathlib import Path
from typing import ClassVar

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter


class DominionTabsCardDownloader(Downloader):
    """Downloads dominiontabs' language-neutral card fields and English
    card text — the two raw files data_refinement needs to build a
    complete, current Dominion card list.

    Single-consumer to src/data_retrieval/dominiontabs/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/dominiontabs")

    CARDS_DB_URL: ClassVar[str] = (
        "https://raw.githubusercontent.com/sumpfork/dominiontabs"
        "/master/card_db_src/cards_db.json"
    )
    CARDS_EN_US_URL: ClassVar[str] = (
        "https://raw.githubusercontent.com/sumpfork/dominiontabs"
        "/master/card_db_src/en_us/cards_en_us.json"
    )

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1() is called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)

    def _run_phase_1(self) -> Path:
        """Download both of dominiontabs' source files verbatim into
        raw_data_dir — no phase_2(): neither file depends on the
        other's content, so both fetches belong in this single
        self-contained phase (see Downloader's own docstring on when a
        source has nothing further to fetch).

        Inputs: none (uses self.raw_data_dir, self.rate_limiter,
            CARDS_DB_URL, CARDS_EN_US_URL).
        Output: raw_data_dir itself — both downloaded files land inside
            it, there's no single combined output file (same
            convention as IsotropicGameLogDownloader's multi-file
            phase_1()).
        Side effects: one paced network request per source file;
            creates raw_data_dir if missing; writes cards_db.json and
            cards_en_us.json into raw_data_dir, overwriting any
            existing copies.
        Exceptions: raises the failing download's
            requests.RequestException if either file fails after
            download_to_file()'s own retries — both files are required
            for this source to be usable at all, unlike a per-item
            best-effort source with many independent files where one
            bad item shouldn't cost the rest.

        Example:
            >>> downloader = DominionTabsCardDownloader()
            >>> files_dir = downloader.phase_1()
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        source_files = [
            (self.CARDS_DB_URL, "cards_db.json"),
            (self.CARDS_EN_US_URL, "cards_en_us.json"),
        ]

        # Fetch each of this source's raw files, verbatim
        for source_url, filename in source_files:
            self.rate_limiter.wait()
            destination_path = self.raw_data_dir / filename
            download_to_file(source_url, destination_path)

        return self.raw_data_dir
