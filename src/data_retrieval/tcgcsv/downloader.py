"""Downloads TCGplayer's card catalog, one game (TCGplayer category) at a
time, from tcgcsv.com's daily JSON mirror.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Reading a catalog into
cards is src/data_refinement/card_binder/tcgcsv/'s job.

SOURCE AND TERMS (read 2026-11-07): tcgcsv.com republishes TCGplayer's
public catalog endpoints once a day as static JSON, one file per category
(game), per group (set) and per group's products; robots.txt allows
everything. The data is TCGplayer's catalog; use here is personal,
noncommercial research. It covers games tcgjson (../tcgjson/) does not
publish, such as Gundam and both Dragon Ball games.

LAYOUT:
    <base>/last-updated.txt      the day's update stamp,
                                 "2026-11-06T20:05:28+0000"
    <base>/tcgplayer/<category id>/groups
    <base>/tcgplayer/<category id>/<group id>/products
Each JSON endpoint answers {success, errors, totalItems, results[]}.

    phase_1() reads the update stamp, then lands each game's catalog in
    <raw_data_dir>/<stamp as YYYYMMDDTHHMMSSZ>/<slug>.json.gz as
    {"category_id", "groups": [...], "products": [...]} (the endpoints'
    results, unchanged), skipping a game whose file is already there.
    The stamp is written last, as last-updated.txt, so a snapshot
    directory holding it is complete for the slugs that run landed
    (latest_snapshot_dir() reads only complete ones; a run with fewer
    slugs than another still marks its directory complete). No phase_2.

One GET per group, so a run is about 210 requests for the four games
here (~4 min at 60/min). tcgcsv updates once a day (around 20:00 UTC); a
run that straddles an update can mix two days' products, which only
matters for cards released that day.
"""

import gzip
import json
import logging
import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from tqdm import tqdm

from src.schema.json_checks import require_int
from src.data_retrieval.download_utils import (
    RESEARCH_USER_AGENT,
    download_to_string,
    latest_complete_dir,
)
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_BASE_URL = "https://tcgcsv.com"
_STAMP_NAME = "last-updated.txt"
# The update stamp as tcgcsv writes it, and the directory name made of it
_STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\+0000")
_SNAPSHOT_NAME = re.compile(r"\d{8}T\d{6}Z")
_HEADERS = {"User-Agent": RESEARCH_USER_AGENT}

# A landed catalog's file name: <slug><CATALOG_FILE_SUFFIX>
CATALOG_FILE_SUFFIX = ".json.gz"

# The games this project downloads, by TCGplayer category id; add one
# when its card stage is built
CATEGORY_IDS: Mapping[str, int] = MappingProxyType(
    {
        "gundam": 86,
        "dragon-ball-fusion-world": 80,
        "dragon-ball-masters": 27,
        "battle-spirits-saga": 72,
    }
)


class TcgcsvCatalogDownloader(Downloader):
    """Lands the day's tcgcsv catalog for each slug.

    Single-consumer to src/data_refinement/card_binder/tcgcsv/ (the
    catalog files) and scripts/run_card_binder_ingestion.py
    (latest_snapshot_dir()).
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/tcgcsv")

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        slugs: tuple[str, ...] = tuple(CATEGORY_IDS),
        base_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter, raw_data_dir: see Downloader.__init__.
            slugs: the games to land, keys of CATEGORY_IDS.
            base_url: tcgcsv's root, no trailing slash; defaults to
                https://tcgcsv.com.
        Output: none. Side effects: none; no I/O until phase_1().
        Exceptions: ValueError if slugs is empty or names a game not in
            CATEGORY_IDS.
        """
        super().__init__(rate_limiter, raw_data_dir)
        unknown = [slug for slug in slugs if slug not in CATEGORY_IDS]
        if not slugs or unknown:
            raise ValueError(f"slugs must be keys of CATEGORY_IDS; got {slugs}")
        self.slugs = slugs
        self.base_url = base_url or _BASE_URL

    @classmethod
    def latest_snapshot_dir(cls, raw_data_dir: Path | None = None) -> Path:
        """The newest complete snapshot directory under raw_data_dir (one
        named YYYYMMDDTHHMMSSZ and holding last-updated.txt); that name
        format sorts by date as a string.

        Inputs: raw_data_dir (DEFAULT_RAW_DATA_DIR when None).
        Output: <raw_data_dir>/<snapshot name>.
        Side effects: lists raw_data_dir.
        Exceptions: FileNotFoundError if no complete snapshot is landed.

        Example:
            >>> TcgcsvCatalogDownloader.latest_snapshot_dir()
            WindowsPath('data/raw/tcgcsv/20261106T200528Z')
        """
        root = cls.DEFAULT_RAW_DATA_DIR if raw_data_dir is None else raw_data_dir
        return latest_complete_dir(root, _SNAPSHOT_NAME, _STAMP_NAME)

    def _run_phase_1(self) -> Path:
        """Land each slug's catalog, then the update stamp.

        Inputs: none.
        Output: the snapshot directory.
        Side effects: one paced GET for the stamp, then per game not yet
            landed one for its groups and one per group; writes the
            catalogs, then last-updated.txt; a tqdm bar per game.
        Exceptions: requests.RequestException after retries; ValueError
            if the stamp or an endpoint's answer is malformed; TypeError
            if a group lacks an int groupId.

        Example:
            >>> TcgcsvCatalogDownloader().phase_1()
            WindowsPath('data/raw/tcgcsv/20261106T200528Z')
        """
        # Name the snapshot after the day's update stamp
        stamp = self._get_text(f"{self.base_url}/{_STAMP_NAME}")
        snapshot_dir = self.raw_data_dir / snapshot_name(stamp)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Land each game not landed yet
        for slug in self.slugs:
            path = snapshot_dir / f"{slug}{CATALOG_FILE_SUFFIX}"
            if path.is_file():
                _logger.info("%s already landed", path)
                continue
            self._land_category(CATEGORY_IDS[slug], path)

        # The stamp last: it marks the snapshot directory complete
        (snapshot_dir / _STAMP_NAME).write_text(stamp, encoding="utf-8")
        _logger.info("tcgcsv snapshot %s: %s", snapshot_dir.name, self.slugs)
        return snapshot_dir

    def _land_category(self, category_id: int, path: Path) -> None:
        """Fetch one category's groups and every group's products, and
        write them to path as one gzipped JSON object.
        Inputs: category_id, path (<snapshot>/<slug>.json.gz).
        Output: none.
        Side effects: 1 + (number of groups) paced GETs; writes path via
            a temporary file, so a half-written catalog is never left at
            path; a tqdm bar.
        Exceptions: requests.RequestException after retries; ValueError
            if an answer is malformed; TypeError if a group lacks an int
            groupId."""
        base = f"{self.base_url}/tcgplayer/{category_id}"
        groups = self._get_results(f"{base}/groups")
        products: list[Any] = []
        for group in tqdm(groups, desc=f"tcgcsv {path.name}", unit="group"):
            if not isinstance(group, dict):
                raise ValueError(f"tcgcsv group is not an object: {group!r}")
            group_id = require_int(group.get("groupId"))
            products.extend(self._get_results(f"{base}/{group_id}/products"))

        catalog = {"category_id": category_id, "groups": groups, "products": products}
        _write_gzip_json_atomically(path, catalog)

    def _get_results(self, url: str) -> list[Any]:
        """One JSON endpoint's results list (parse_results).
        Inputs: url. Output: list of raw JSON values.
        Side effects: one paced GET.
        Exceptions: requests.RequestException after retries;
            ValueError (parse_results)."""
        return parse_results(self._get_text(url), url)

    def _get_text(self, url: str) -> str:
        """One paced GET, with the research User-Agent.
        Inputs: url. Output: the body as text.
        Side effects: one paced GET.
        Exceptions: requests.RequestException after retries."""
        self.rate_limiter.wait()
        return download_to_string(url, headers=_HEADERS)


def snapshot_name(stamp: str) -> str:
    """The snapshot directory's name for one update stamp.

    Inputs: stamp (last-updated.txt's text, surrounding whitespace
        ignored).
    Output: "YYYYMMDDTHHMMSSZ".
    Side effects: none.
    Exceptions: ValueError if stamp is not "YYYY-MM-DDTHH:MM:SS+0000".

    Example:
        >>> snapshot_name("2026-11-06T20:05:28+0000")
        '20261106T200528Z'
    """
    match = _STAMP.fullmatch(stamp.strip())
    if match is None:
        raise ValueError(f"tcgcsv update stamp is {stamp!r}")
    return "{}{}{}T{}{}{}Z".format(*match.groups())


def parse_results(text: str, url: str) -> list[Any]:
    """An endpoint answer's results list.

    Inputs: text (the answer's JSON), url (named in errors).
    Output: results, unchanged.
    Side effects: none.
    Exceptions: ValueError if text is not a JSON object, success is not
        true, or results is not a list.

    Example:
        >>> parse_results('{"success": true, "errors": [], "results": [1]}', u)
        [1]
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{url}: not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{url}: not a JSON object")
    if payload.get("success") is not True:
        raise ValueError(f"{url}: success is not true: {payload.get('errors')!r}")
    result = payload.get("results")
    if not isinstance(result, list):
        raise ValueError(f"{url}: results is not a list")
    return result


def _write_gzip_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write payload to path as gzipped JSON via path.tmp and os.replace,
    so path never holds a part-written catalog.
    Inputs: path, payload. Output: none.
    Side effects: writes path.tmp, then replaces path.
    Exceptions: OSError on a write or replace failure."""
    temp_path = path.with_name(path.name + ".tmp")
    with gzip.open(temp_path, "wt", encoding="utf-8") as file:
        json.dump(payload, file)
    os.replace(temp_path, path)
