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
from typing import ClassVar, Tuple

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

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


class PokemonTcgDataDownloader(Downloader):
    """Downloads and extracts the pokemon-tcg-data repo's card and deck JSON.

    Single-consumer to src/data_retrieval/pokemon_tcg/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/pokemon_tcg")

    def __init__(
        self,
        repo_zip_url: str,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
    ) -> None:
        """
        Inputs:
            repo_zip_url: GitHub zipball URL for the repo (e.g.
                "https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/zipball").
                Required and positional: no sensible default.
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
        Output: none (constructor).
        Side effects: none — no I/O happens until phase_1()/download()/
            extract() are called.
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.repo_zip_url = repo_zip_url

    def _run_phase_1(self) -> Path:
        """Download the repo zip, then extract it, returning
        raw_data_dir — the extraction produces two equally-important
        sibling directories (cards/, decks/), so there's no single
        primary artifact to return instead.

        Composed of download() followed by extract() — no additional
        logic beyond calling the two. extract()'s own richer
        PokemonTcgDataDownloadResult (both directory paths) stays
        available via calling download()/extract() directly.

        Inputs: none (uses self.repo_zip_url, self.raw_data_dir).
        Output: self.raw_data_dir.
        Side effects: one network request; writes many files to
            raw_data_dir.
        Exceptions: whatever download() or extract() raise.

        Example:
            >>> downloader = PokemonTcgDataDownloader(
            ...     "https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/zipball",
            ... )
            >>> raw_data_dir = downloader.phase_1()
        """
        zip_path = self.download()
        self.extract(zip_path)
        return self.raw_data_dir

    def download(self) -> Path:
        """Download the repo zipball into self.raw_data_dir, unmodified.

        Inputs: none (uses self.repo_zip_url, self.raw_data_dir,
            self.rate_limiter).
        Output: path to the saved .zip file.
        Side effects: one paced network request; creates raw_data_dir
            if missing; writes one file to disk.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response) or on failure to write the file. Any
            partially-written file is removed before the exception
            propagates.
        """
        destination_path = self.raw_data_dir / _ARCHIVE_FILENAME

        self.rate_limiter.wait()
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
            if member.is_dir() or not self._member_matches_suffix(member, suffix_parts):
                continue
            matched_any = True
            destination_dir.mkdir(parents=True, exist_ok=True)
            self._extract_one_member(zip_file, member, destination_dir)

        if not matched_any:
            raise ValueError(
                f"No zip members found matching path suffix {path_suffix!r} "
                f"in {zip_file.filename!r}"
            )

        return destination_dir

    @staticmethod
    def _member_matches_suffix(
        member: zipfile.ZipInfo, suffix_parts: Tuple[str, ...]
    ) -> bool:
        """Whether member's path (minus the zipball's top-level repo folder)
        is exactly suffix_parts plus one trailing *.json filename.

        Inputs: member, the zip entry to check; suffix_parts, the target
            directory path split into parts (e.g. ("cards", "en")).
        Output: bool.
        Side effects: none.
        Exceptions: none.
        """
        member_parts = PurePosixPath(member.filename).parts[1:]
        return (
            len(member_parts) == len(suffix_parts) + 1
            and member_parts[:-1] == suffix_parts
            and member_parts[-1].endswith(".json")
        )

    @staticmethod
    def _extract_one_member(
        zip_file: zipfile.ZipFile, member: zipfile.ZipInfo, destination_dir: Path
    ) -> None:
        """Copy one matched zip member into destination_dir, flattened to
        just its filename.

        Inputs: zip_file, the open archive; member, the entry to copy;
            destination_dir, already created by the caller.
        Output: none.
        Side effects: writes destination_dir/<member's filename>.
        Exceptions: raises on a write failure, after removing the partial
            destination file (see extract()'s docstring on why earlier
            members' files are not also rolled back).
        """
        destination_path = destination_dir / PurePosixPath(member.filename).name
        try:
            with zip_file.open(member) as source_file:
                with open(destination_path, "wb") as destination_file:
                    shutil.copyfileobj(source_file, destination_file)
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise
