"""Downloads the pokemon-tcg-data community GitHub repo.

V1 implementation. Source: github.com/PokemonTCG/pokemon-tcg-data — a
community-maintained repo of card and deck JSON, roughly one year
stale at the time this was written. If the official Pokemon TCG V2 API
turns out to be a better source, that would land as a separate V2
downloader, not a rewrite of this one.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).
"""

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar

from src.data_retrieval.download_utils import download_to_file

_ARCHIVE_FILENAME = "pokemon-tcg-data.zip"  # zipball URLs carry no filename
# of their own to derive one from
_CARDS_PATH_SUFFIX = "cards/en"
_DECKS_PATH_SUFFIX = "decks/en"


@dataclass
class PokemonTcgDataDownloadResult:
    """Paths to the two directories a completed fetch leaves on disk.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    cards_dir: Path  # extracted *.json files from the repo's cards/en/
    decks_dir: Path  # extracted *.json files from the repo's decks/en/


class PokemonTcgDataDownloader:
    """Downloads and extracts the pokemon-tcg-data repo's card and deck JSON.

    Single-consumer to src/data_retrieval/pokemon_tcg/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/pokemon_tcg")

    def __init__(self, repo_zip_url: str, raw_data_dir: Path | None = None) -> None:
        """
        Inputs:
            repo_zip_url: GitHub zipball URL for the repo (e.g.
                "https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/zipball").
            raw_data_dir: directory the extracted cards/ and decks/
                subdirectories are written into. Defaults to
                DEFAULT_RAW_DATA_DIR when omitted (expected to be a
                path under data/raw, per src/README.md — not this
                class's concern to enforce, just to receive).
        Output: none (constructor).
        Side effects: none — no I/O happens until fetch()/download()/
            extract() are called.
        Exceptions: none.
        """
        self.repo_zip_url = repo_zip_url
        self.raw_data_dir = (
            raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR
        )

    def fetch(self) -> PokemonTcgDataDownloadResult:
        """Entry point: download the repo zip, then extract it.

        Composed of download() followed by extract() — no additional
        logic beyond calling the two and returning their result.

        Inputs: none (uses self.repo_zip_url, self.raw_data_dir).
        Output: PokemonTcgDataDownloadResult with both resulting
            directory paths.
        Side effects: one network request; writes many files to
            raw_data_dir.
        Exceptions: whatever download() or extract() raise.

        Example:
            >>> downloader = PokemonTcgDataDownloader(
            ...     "https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/zipball",
            ... )
            >>> result = downloader.fetch()
        """
        zip_path = self.download()
        return self.extract(zip_path)

    def download(self) -> Path:
        """Download the repo zipball into self.raw_data_dir, unmodified.

        Inputs: none (uses self.repo_zip_url, self.raw_data_dir).
        Output: path to the saved .zip file.
        Side effects: one network request; creates raw_data_dir if
            missing; writes one file to disk.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file. Any
            partially-written file is removed before the exception
            propagates.
        """
        destination_path = self.raw_data_dir / _ARCHIVE_FILENAME

        download_to_file(self.repo_zip_url, destination_path)

        return destination_path

    def extract(self, zip_path: Path) -> PokemonTcgDataDownloadResult:
        """Extract cards/en/*.json and decks/en/*.json from the zipball.

        A GitHub zipball wraps the whole repo under one variable,
        commit-specific top-level folder (e.g.
        "PokemonTCG-pokemon-tcg-data-<sha>/"), so members are matched
        by their path *suffix* (".../cards/en/<file>.json", ".../decks/en/<file>.json"),
        not an exact path. Delegates the actual per-directory copy to
        _extract_matching_members so the cards/en and decks/en cases
        share one implementation rather than two near-identical ones.

        Inputs:
            zip_path: path to a .zip file (e.g. one returned by
                download()).
        Output: PokemonTcgDataDownloadResult with both extracted
            directory paths.
        Side effects: writes one .json file per matched zip member,
            into raw_data_dir/cards/ and raw_data_dir/decks/.
        Exceptions: raises if zip_path doesn't exist, isn't a valid
            zip file, no members match a given prefix (e.g. the repo's
            layout changed since this was written), or a file can't be
            written. On a mid-extraction failure, only the single file
            being written at the time is cleaned up — files already
            written earlier in the call are left in place, since
            re-running extract() is idempotent per file (see
            _extract_matching_members).
        """
        with zipfile.ZipFile(zip_path) as zip_file:
            cards_dir = self._extract_matching_members(
                zip_file, _CARDS_PATH_SUFFIX, self.raw_data_dir / "cards"
            )
            decks_dir = self._extract_matching_members(
                zip_file, _DECKS_PATH_SUFFIX, self.raw_data_dir / "decks"
            )

        return PokemonTcgDataDownloadResult(cards_dir=cards_dir, decks_dir=decks_dir)

    def _extract_matching_members(
        self, zip_file: zipfile.ZipFile, path_suffix: str, destination_dir: Path
    ) -> Path:
        """Copy every zip member whose path ends with path_suffix/<name>.json.

        Private helper — single consumer is extract(), called once per
        top-level directory of interest (cards/en, decks/en).

        Inputs:
            zip_file: an already-open ZipFile to read members from.
            path_suffix: the path suffix to match members against,
                e.g. "cards/en" or "decks/en".
            destination_dir: directory the matched files are copied
                into, flattened (original repo directory structure
                above path_suffix is not preserved).
        Output: destination_dir, for convenience chaining in extract().
        Side effects: creates destination_dir if missing; writes one
            file per matched member.
        Exceptions: raises if no members match path_suffix, or a file
            can't be written. The single file being written when the
            failure happens is removed before the exception propagates
            (same convention as download()). Files from earlier
            members in this same call are deliberately *not* rolled
            back — re-running extract() is idempotent (every matched
            file is overwritten), so a partially-populated
            destination_dir left by a failed run is safe to resume
            from rather than needing atomic all-or-nothing extraction.
        """
        suffix_parts = tuple(path_suffix.split("/"))
        matched_any = False

        for member in zip_file.infolist():
            if member.is_dir():
                continue

            # Strip the zipball's variable top-level repo folder before matching.
            member_parts = PurePosixPath(member.filename).parts[1:]
            if (
                len(member_parts) != len(suffix_parts) + 1
                or member_parts[:-1] != suffix_parts
                or not member_parts[-1].endswith(".json")
            ):
                continue

            matched_any = True
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / member_parts[-1]
            try:
                with zip_file.open(member) as source_file:
                    with open(destination_path, "wb") as destination_file:
                        shutil.copyfileobj(source_file, destination_file)
            except Exception:
                destination_path.unlink(missing_ok=True)
                raise

        if not matched_any:
            raise ValueError(
                f"No zip members found matching path suffix {path_suffix!r} "
                f"in {zip_file.filename!r}"
            )

        return destination_dir
