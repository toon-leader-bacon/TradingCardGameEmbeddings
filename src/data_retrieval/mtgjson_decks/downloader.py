"""Downloads MTGJSON's AllDeckFiles: one JSON file per Wizards-printed
precon (theme, intro, Commander, duel, Jumpstart, starter and other
decks, 1993 to today).

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Which deck types are
playable decks (and which, like Secret Lair drops, are not) is
data_refinement's call.

THE SOURCE: https://mtgjson.com/downloads/all-files/ (MIT licence).
MTGJSON rebuilds every file daily; Meta.json names the build
("5.3.0+20261105"). robots.txt disallows crawlers on /api/v5/*.json,
which keeps search engines off the files; the download page links them
for direct use, so this downloader takes the one bulk zip (~265 MB) and
its published SHA-256, three requests per build.

CONFIRMED LIVE 2026-11-05. phase_1() reads Meta.json and lands that
build under <raw_data_dir>/<build>/: decks/ (the zip's JSON files) and
Meta.json, the latter written last, so a build directory with Meta.json
is complete and a re-run on the same build downloads nothing. The zip
is deleted once extracted. There is no phase 2.
"""

import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import ClassVar

from src.data_retrieval.download_utils import download_to_file, download_to_string
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_API_URL = "https://mtgjson.com/api/v5"
_ZIP_NAME = "AllDeckFiles.zip"
_META_NAME = "Meta.json"
_DECKS_DIR_NAME = "decks"
_DOWNLOAD_TIMEOUT_SECONDS = 600


class MtgjsonDeckFilesDownloader(Downloader):
    """Downloads and extracts MTGJSON's AllDeckFiles.zip, one directory
    per MTGJSON build.

    Single-consumer to src/data_refinement/deck_box/mtgjson_decks/.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/mtgjson_decks")

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        api_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
            api_url: MTGJSON's file root, no trailing slash (a file is
                <api_url>/<name>). Defaults to mtgjson.com/api/v5.
        Output: none (constructor).
        Side effects: none; no I/O until phase_1().
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.api_url = api_url or _API_URL

    @classmethod
    def latest_build_decks_dir(cls, raw_data_dir: Path | None = None) -> Path:
        """The decks/ directory of the newest complete build landed under
        raw_data_dir (a build directory holding Meta.json).

        Newest by the build date after "+" ("5.3.0+20261105"), not by
        the whole name: a string sort would put "5.10.0+..." before
        "5.3.0+...". A directory whose name has no "+<yyyymmdd>" is
        not a build and is skipped; two complete builds of one date are
        ordered by name.

        Inputs: raw_data_dir (DEFAULT_RAW_DATA_DIR when None).
        Output: <raw_data_dir>/<build>/decks.
        Side effects: lists raw_data_dir.
        Exceptions: FileNotFoundError if no complete build is landed.

        Example:
            >>> MtgjsonDeckFilesDownloader.latest_build_decks_dir()
            PosixPath('data/raw/mtgjson_decks/5.3.0+20261105/decks')
        """
        raise NotImplementedError

    def _run_phase_1(self) -> Path:
        """Land the current build's deck files, unless already landed.

        Inputs: none.
        Output: the build directory, <raw_data_dir>/<build>/, holding
            decks/*.json and Meta.json.
        Side effects: up to four paced requests (Meta.json, the
            .sha256, the zip, Meta.json again); writes the build
            directory; deletes the zip after extracting it.
        Exceptions: requests.RequestException if a request still fails
            after retries; ValueError if Meta.json has no build version,
            the .sha256 holds no digest, the zip's SHA-256 is wrong (the
            zip is deleted first), or MTGJSON rebuilt mid-download (the
            zip and digest may be the new build's; re-run);
            zipfile.BadZipFile for an unreadable zip; OSError on a
            failed write.

        Example:
            >>> MtgjsonDeckFilesDownloader().phase_1()
            PosixPath('data/raw/mtgjson_decks/5.3.0+20261105')
        """
        # Which build is current
        self.rate_limiter.wait()
        meta_text = download_to_string(f"{self.api_url}/{_META_NAME}")
        build_dir = self.raw_data_dir / _build_version_of(meta_text)
        if (build_dir / _META_NAME).is_file():
            _logger.info("MTGJSON build already landed: %s", build_dir)
            return build_dir

        # Download and check the zip
        zip_path = build_dir / _ZIP_NAME
        self.rate_limiter.wait()
        expected_sha256 = _sha256_in_checksum_file(
            download_to_string(f"{self.api_url}/{_ZIP_NAME}.sha256")
        )
        self.rate_limiter.wait()
        download_to_file(
            f"{self.api_url}/{_ZIP_NAME}",
            zip_path,
            timeout_seconds=_DOWNLOAD_TIMEOUT_SECONDS,
        )
        _check_sha256(zip_path, expected_sha256)
        # A rebuild mid-download would land the new zip under the old label
        self.rate_limiter.wait()
        if download_to_string(f"{self.api_url}/{_META_NAME}") != meta_text:
            raise ValueError("MTGJSON rebuilt during the download; re-run")

        # Unwrap; Meta.json last marks the build complete
        _extract_json_members(zip_path, build_dir / _DECKS_DIR_NAME)
        zip_path.unlink()
        (build_dir / _META_NAME).write_text(meta_text, encoding="utf-8")
        return build_dir


def _build_version_of(meta_text: str) -> str:
    """The build version in a Meta.json body, e.g. "5.3.0+20261105".

    Inputs: meta_text (Meta.json's body). Output: the version string.
    Side effects: none.
    Exceptions: ValueError if the body isn't JSON with a non-empty
        string at meta.version, or the version holds a path separator.
    """
    try:
        version = json.loads(meta_text)["meta"]["version"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Meta.json has no meta.version: {meta_text[:200]!r}"
        ) from error
    if not isinstance(version, str) or not version or "/" in version or "\\" in version:
        raise ValueError(f"Meta.json meta.version is not a plain name: {version!r}")
    return version


def _sha256_in_checksum_file(checksum_text: str) -> str:
    """The digest in a .sha256 body ("<hex>" or "<hex>  <file name>").

    Inputs: checksum_text. Output: 64 lowercase hex characters.
    Side effects: none.
    Exceptions: ValueError if the first token is not a SHA-256 digest
        (an empty body, or an HTML error page served with 200).
    """
    tokens = checksum_text.split()
    digest = tokens[0].lower() if tokens else ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f".sha256 holds no SHA-256 digest: {checksum_text[:200]!r}")
    return digest


def _extract_json_members(zip_path: Path, destination_dir: Path) -> None:
    """Write every *.json member of the zip into destination_dir, flat
    (by base name; MTGJSON's deck file names are unique).

    Inputs: zip_path, destination_dir. Output: none.
    Side effects: creates destination_dir; writes one file per member,
        overwriting any of the same name (so a re-run after a failure is
        safe).
    Exceptions: zipfile.BadZipFile; ValueError if the zip has no JSON
        members, or two members share a base name.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    with zipfile.ZipFile(zip_path) as zip_file:
        for member in zip_file.infolist():
            # Base name only, so no member can write outside destination_dir
            name = Path(member.filename).name
            if member.is_dir() or not name.endswith(".json"):
                continue
            if name in written:
                raise ValueError(f"{zip_path.name}: two members named {name}")
            (destination_dir / name).write_bytes(zip_file.read(member))
            written.add(name)
    if not written:
        raise ValueError(f"{zip_path.name} holds no JSON files")
    _logger.info("Extracted %d deck files into %s", len(written), destination_dir)


def _check_sha256(path: Path, expected_sha256: str) -> None:
    """Delete path and raise if its SHA-256 isn't expected_sha256.

    Inputs: path, expected_sha256 (lowercase hex). Output: none.
    Side effects: reads path; deletes it on a mismatch.
    Exceptions: ValueError on a mismatch; OSError if path can't be read.
    """
    with open(path, "rb") as file:
        actual_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        path.unlink()
        raise ValueError(
            f"{path.name}: SHA-256 {actual_sha256}, expected {expected_sha256}; deleted"
        )
