"""Downloads tcgjson's weekly card catalogs (TCGplayer's singles catalog,
one gzipped JSON per game) from the latest GitHub release.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Reading a catalog into
cards is src/data_refinement/card_binder/tcgjson/'s job.

SOURCE AND TERMS (read 2026-11-06): tcgjson (github.com/HanClinto/tcgjson,
MIT-licensed code) scrapes TCGplayer's public catalog endpoints once a
week and publishes the result as GitHub Release assets; this downloader
fetches only those assets, never TCGplayer itself. TCGplayer's robots.txt
has no AI clause and allows product pages. The data is TCGplayer's
catalog; use here is personal, noncommercial research.

LAYOUT: every release carries bulk-data.json, a Scryfall-style manifest
whose data[] entries name each file (type "<slug>_catalog_full" ->
"<slug>.full.json.gz") with its size and SHA-256, plus the release's
generated_at. The latest release is always at
<_RELEASE_URL>/<file name>.

    phase_1() reads the manifest and finds every slug's full catalog in
    it (a missing slug raises before anything is downloaded), then lands
    each catalog in <raw_data_dir>/<generated_at as YYYYMMDDTHHMMSSZ>/,
    skipping a file already there with the right SHA-256 and checking
    each download's size and SHA-256. The manifest is written last, so a
    release directory holding bulk-data.json is complete
    (latest_release_dir() reads only complete ones). No phase_2.

The manifest is downloaded data, so every name taken from it into a path
is checked first: the release name must be YYYYMMDDTHHMMSSZ, a file name
a plain name with no separators.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Mapping

from src.data_retrieval.download_utils import (
    check_sha256,
    sha256_of,
    download_to_file,
    download_to_string,
)
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_RELEASE_URL = "https://github.com/HanClinto/tcgjson/releases/latest/download"
_MANIFEST_NAME = "bulk-data.json"
_FULL_CATALOG_TYPE = "{slug}_catalog_full"
# generated_at as the manifest writes it, and the directory name made of it
_GENERATED_AT = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z")
_RELEASE_NAME = re.compile(r"\d{8}T\d{6}Z")
# A release asset's name: no separators, so it can't leave the release dir
_ASSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

# The games this project downloads; add a slug when its card stage is built
CATALOG_SLUGS: tuple[str, ...] = (
    "lorcana",
    "yugioh",
    "digimon-card-game",
    "one-piece",
    "star-wars-unlimited",
    "union-arena",
    "riftbound",
)


@dataclass(frozen=True)
class CatalogFile:
    """One file a release's manifest lists.

    file_name: the asset's name in the release ("lorcana.full.json.gz"),
        checked to be a plain name.
    size: its size in bytes. sha256: its lowercase hex SHA-256.
    """

    file_name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseManifest:
    """The parts of bulk-data.json this downloader reads.

    release_name: generated_at as YYYYMMDDTHHMMSSZ, the release
        directory's name.
    files: every listed file, by its manifest type
        ("lorcana_catalog_full").
    """

    release_name: str
    files: Mapping[str, CatalogFile]


class TcgjsonCatalogDownloader(Downloader):
    """Lands the latest tcgjson release's full catalog for each slug.

    Single-consumer to src/data_refinement/card_binder/tcgjson/ (the
    catalog files) and scripts/run_card_binder_ingestion.py
    (latest_release_dir()).
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/tcgjson")

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        slugs: tuple[str, ...] = CATALOG_SLUGS,
        release_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter, raw_data_dir: see Downloader.__init__.
            slugs: the games whose full catalog to land (tcgjson's file
                stems, e.g. "lorcana").
            release_url: the release's file root, no trailing slash;
                defaults to the latest release.
        Output: none. Side effects: none; no I/O until phase_1().
        Exceptions: ValueError if slugs is empty.
        """
        super().__init__(rate_limiter, raw_data_dir)
        if not slugs:
            raise ValueError("slugs must name at least one catalog")
        self.slugs = slugs
        self.release_url = release_url or _RELEASE_URL

    @classmethod
    def latest_release_dir(cls, raw_data_dir: Path | None = None) -> Path:
        """The newest complete release directory under raw_data_dir (one
        named YYYYMMDDTHHMMSSZ and holding bulk-data.json); that name
        format sorts by date as a string.

        Inputs: raw_data_dir (DEFAULT_RAW_DATA_DIR when None).
        Output: <raw_data_dir>/<release name>.
        Side effects: lists raw_data_dir.
        Exceptions: FileNotFoundError if no complete release is landed.

        Example:
            >>> TcgjsonCatalogDownloader.latest_release_dir()
            PosixPath('data/raw/tcgjson/20260831T174333Z')
        """
        root = cls.DEFAULT_RAW_DATA_DIR if raw_data_dir is None else raw_data_dir
        result: list[Path] = []
        if root.is_dir():
            result = sorted(
                path
                for path in root.iterdir()
                if _RELEASE_NAME.fullmatch(path.name)
                and (path / _MANIFEST_NAME).is_file()
            )
        if not result:
            raise FileNotFoundError(f"no complete tcgjson release under {root}")
        return result[-1]

    def _run_phase_1(self) -> Path:
        """Land each slug's full catalog, then the manifest.

        Inputs: none.
        Output: the release directory.
        Side effects: one paced GET for the manifest and one per catalog
            not already landed; writes the catalogs, then bulk-data.json.
        Exceptions: requests.RequestException after retries; ValueError
            if the manifest is malformed, lists no full catalog for a slug
            (raised before any download), or a download's size or SHA-256
            is wrong (the file is deleted).

        Example:
            >>> TcgjsonCatalogDownloader().phase_1()
            PosixPath('data/raw/tcgjson/20260831T174333Z')
        """
        # Read the manifest; find every slug's catalog before downloading
        self.rate_limiter.wait()
        manifest_text = download_to_string(f"{self.release_url}/{_MANIFEST_NAME}")
        manifest = parse_release_manifest(manifest_text)
        catalog_files = self._catalog_files(manifest)
        release_dir = self.raw_data_dir / manifest.release_name
        release_dir.mkdir(parents=True, exist_ok=True)

        # Land each slug's catalog
        for catalog_file in catalog_files:
            self._land_catalog(catalog_file, release_dir)

        # The manifest last: it marks the release directory complete
        (release_dir / _MANIFEST_NAME).write_text(manifest_text, encoding="utf-8")
        _logger.info("tcgjson release %s: %s", manifest.release_name, self.slugs)
        return release_dir

    def _catalog_files(self, manifest: ReleaseManifest) -> list[CatalogFile]:
        """Each slug's full catalog entry, in slug order.

        Inputs: manifest. Output: list of CatalogFile.
        Side effects: none.
        Exceptions: ValueError naming every slug the manifest lacks."""
        kinds = {slug: _FULL_CATALOG_TYPE.format(slug=slug) for slug in self.slugs}
        missing = [slug for slug, kind in kinds.items() if kind not in manifest.files]
        if missing:
            raise ValueError(f"tcgjson manifest has no full catalog for {missing}")
        return [manifest.files[kind] for kind in kinds.values()]

    def _land_catalog(self, catalog_file: CatalogFile, release_dir: Path) -> None:
        """Download one catalog into release_dir unless it is already
        there with the right SHA-256; check the download's size and
        SHA-256.

        Inputs: catalog_file, release_dir.
        Output: none.
        Side effects: at most one paced GET; writes one file.
        Exceptions: requests.RequestException after retries; ValueError
            on a size or SHA-256 mismatch (the file is deleted)."""
        path = release_dir / catalog_file.file_name
        if path.is_file() and sha256_of(path) == catalog_file.sha256:
            _logger.info("%s already landed", path)
            return

        self.rate_limiter.wait()
        download_to_file(f"{self.release_url}/{catalog_file.file_name}", path)
        size = path.stat().st_size
        if size != catalog_file.size:
            path.unlink()
            raise ValueError(
                f"{path.name}: {size} bytes, expected {catalog_file.size}; deleted"
            )
        check_sha256(path, catalog_file.sha256)


def parse_release_manifest(manifest_text: str) -> ReleaseManifest:
    """Parse bulk-data.json into a ReleaseManifest.

    Inputs: manifest_text (the manifest's JSON).
    Output: ReleaseManifest; release_name is generated_at
        ("2026-08-31T17:43:33Z") without "-" and ":".
    Side effects: none.
    Exceptions: ValueError if it is not a JSON object, generated_at is
        not "YYYY-MM-DDTHH:MM:SSZ", data is not a list, or an entry lacks
        a string type, a plain file name (no separators) as
        download_uri, an int size or a hex sha256.

    Example:
        >>> parse_release_manifest(text).files["lorcana_catalog_full"].file_name
        'lorcana.full.json.gz'
    """
    try:
        payload = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"tcgjson manifest is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("tcgjson manifest is not a JSON object")

    # The release name, from generated_at
    generated_at = payload.get("generated_at")
    match = (
        _GENERATED_AT.fullmatch(generated_at) if isinstance(generated_at, str) else None
    )
    if match is None:
        raise ValueError(f"tcgjson manifest generated_at is {generated_at!r}")
    release_name = "{}{}{}T{}{}{}Z".format(*match.groups())

    # Every listed file
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise ValueError("tcgjson manifest has no data list")
    files = dict(_parse_manifest_entry(entry) for entry in entries)
    return ReleaseManifest(release_name, files)


def _parse_manifest_entry(entry: object) -> tuple[str, CatalogFile]:
    """One data[] entry as (type, CatalogFile).

    Inputs: entry (a parsed JSON value). Output: (type, CatalogFile).
    Side effects: none.
    Exceptions: ValueError if a field is missing, of the wrong type, or
        download_uri is not a plain file name."""
    if not isinstance(entry, dict):
        raise ValueError(f"tcgjson manifest entry is not an object: {entry!r}")
    kind, file_name = entry.get("type"), entry.get("download_uri")
    size, sha256 = entry.get("size"), entry.get("sha256")
    if not isinstance(kind, str) or not isinstance(file_name, str):
        raise ValueError(f"tcgjson manifest entry lacks type/download_uri: {entry!r}")
    if not _ASSET_NAME.fullmatch(file_name) or ".." in file_name:
        raise ValueError(f"tcgjson manifest names an unsafe file: {file_name!r}")
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"tcgjson manifest entry {kind} has size {size!r}")
    if not isinstance(sha256, str) or not _SHA256_HEX.fullmatch(sha256):
        raise ValueError(f"tcgjson manifest entry {kind} has sha256 {sha256!r}")
    return kind, CatalogFile(file_name, size, sha256)
