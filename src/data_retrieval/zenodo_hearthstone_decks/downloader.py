"""Downloads the Zenodo "Hearthstone Decks Dataset": two CSV files.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering.

THE DATASET: https://zenodo.org/records/10198504 (DOI
10.5281/zenodo.10198504, v1, 2023-11-22, CC BY 4.0), by García-Sánchez,
Tonda, Fernández-Ares and Mora, the data of Mora et al., "Looking for
archetypes: Applying game data mining to Hearthstone decks",
Entertainment Computing 43, 100498 (2022). Cite both when using it.

    decks.csv (56 MB): ";"-separated, one row per deck: idDeck, Minions,
        Spells, Weapons, DeckType, DeckArchetype, CraftingCost, Date
        (m/d/yyyy), Set (the patch). 852,286 decks, 2013-2020.
    cards.csv (564 MB): ";"-separated, one row per card copy: idDeck,
        idCard (the card's English name, sometimes with a trailing
        space).

PROVENANCE: the record doesn't name its source, but the deck ids are
HearthPwn deck ids. HearthPwn's robots.txt now says ai-train=no; the
user chose (2026-10-28) to use this published research dataset for
comparability with the paper. See src/data_retrieval/TODO.md.

CONFIRMED LIVE 2026-10-28. phase_1() downloads each file not already on
disk with the right MD5, then checks its MD5 against the record's, so a
re-run downloads nothing and a corrupt file never stays on disk.
There is no phase 2.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_RECORD_FILES_URL = "https://zenodo.org/records/10198504/files"
_DOWNLOAD_TIMEOUT_SECONDS = 600
_MD5_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class ZenodoFile:
    """One file of the Zenodo record: its name and published MD5."""

    filename: str
    md5: str


class ZenodoHearthstoneDecksDownloader(Downloader):
    """Downloads decks.csv and cards.csv from Zenodo record 10198504.

    Single-consumer to src/data_retrieval/zenodo_hearthstone_decks/.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/zenodo_hearthstone_decks")
    DECKS_FILE: ClassVar[ZenodoFile] = ZenodoFile(
        "decks.csv", "7cf8ec9e379dabce336906a0566184cd"
    )
    CARDS_FILE: ClassVar[ZenodoFile] = ZenodoFile(
        "cards.csv", "4e6bf3c59a225fff80475b8c42ca3d6f"
    )

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        files_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
            files_url: the record's files URL, no trailing slash
                (a file is <files_url>/<filename>). Defaults to Zenodo.
        Output: none (constructor).
        Side effects: none; no I/O until phase_1().
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.files_url = files_url or _RECORD_FILES_URL

    def _run_phase_1(self) -> Path:
        """Download each file not already on disk with its published
        MD5, and check it.

        Inputs: none.
        Output: self.raw_data_dir (holding decks.csv and cards.csv).
        Side effects: up to two paced downloads (~620 MB); creates
            raw_data_dir; replaces a file whose MD5 is wrong.
        Exceptions: requests.RequestException if a download still fails
            after retries; ValueError if a downloaded file's MD5 is
            wrong (the file is deleted first).

        Example:
            >>> ZenodoHearthstoneDecksDownloader().phase_1()
            PosixPath('data/raw/zenodo_hearthstone_decks')
        """
        for zenodo_file in (self.DECKS_FILE, self.CARDS_FILE):
            destination_path = self.raw_data_dir / zenodo_file.filename
            if (
                destination_path.is_file()
                and _md5_of(destination_path) == zenodo_file.md5
            ):
                continue
            self.rate_limiter.wait()
            download_to_file(
                f"{self.files_url}/{zenodo_file.filename}",
                destination_path,
                timeout_seconds=_DOWNLOAD_TIMEOUT_SECONDS,
            )
            _check_md5(destination_path, zenodo_file.md5)
        return self.raw_data_dir


def _check_md5(path: Path, expected_md5: str) -> None:
    """Delete path and raise if its MD5 isn't expected_md5.

    Inputs: path, expected_md5 (lowercase hex). Output: none.
    Side effects: reads path; deletes it on a mismatch.
    Exceptions: ValueError on a mismatch.
    """
    actual_md5 = _md5_of(path)
    if actual_md5 != expected_md5:
        path.unlink()
        raise ValueError(
            f"{path.name}: MD5 {actual_md5}, expected {expected_md5}; deleted"
        )


def _md5_of(path: Path) -> str:
    """path's MD5 as lowercase hex, read in chunks.

    Inputs: path. Output: the hex digest. Side effects: reads path.
    Exceptions: OSError if path can't be read.
    """
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_MD5_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
