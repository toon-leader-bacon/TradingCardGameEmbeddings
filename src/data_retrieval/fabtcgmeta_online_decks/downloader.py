"""Downloads fabtcgmeta.com's online (Talishar) Flesh and Blood decklists.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. See
src/data_retrieval/fab_deck_sources.md for the source survey that picked
this source, and for its robots.txt (fabtcgmeta.com allows AI crawlers).

THE DATA: fabtcgmeta's front end reads static JSON from
https://data.fabtcgmeta.com/v1/. Each distinct deck played on Talishar
is keyed by a 64-hex hash, and decks are sharded by the hash's first
three hex characters, in two tiers:

    decklist/online/archive/<prefix>.json  (~200 decks each, Oct 2025 on)
    decklist/online/recent/<prefix>.json   (~15 decks each, the last weeks)

A shard is one JSON object, {deck hash: deck}, each deck holding its
hero, format, date and card list (see
src/data_refinement/deck_box/fabtcgmeta_online_decks/extraction_stage.py).
The two tiers didn't overlap in the shards sampled.

There is no listing of the shards: the 3-hex prefixes are all 4,096
values, so phase_1() asks for each, paced at 60/min (~2.3 h for 8,192
shards, ~2.5 GB). A shard already on disk is skipped, so a re-run only
fetches what's missing. Both tiers together hold a rolling 365-day
window (decks move from recent into archive and then age out), so every
shard on disk is a snapshot: a refresh must re-fetch both tiers. A 404 (a prefix with no decks)
is counted and logged, not saved, so a later run asks again.

CONFIRMED LIVE 2026-10-28. There is no phase 2.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import ClassVar

import requests
from tqdm import tqdm

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_DATA_URL = "https://data.fabtcgmeta.com/v1/decklist/online"
_SHARD_PREFIX_HEX_DIGITS = 3
_HTTP_NOT_FOUND = 404


class ShardTier(str, Enum):
    """fabtcgmeta's two online-deck shard sets (see the module docstring)."""

    ARCHIVE = "archive"
    RECENT = "recent"


class FabtcgmetaOnlineDecksDownloader(Downloader):
    """Downloads every online-deck shard, archive and recent."""

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/fabtcgmeta_online_decks")

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        data_url: str | None = None,
        shard_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__.
            raw_data_dir: see Downloader.__init__.
            data_url: the online-decklist URL, no trailing slash (a
                shard is <data_url>/<tier>/<prefix>.json). Defaults to
                fabtcgmeta's.
            shard_prefixes: the prefixes to fetch. Defaults to every
                3-hex prefix (000-fff); tests pass a few.
        Output: none (constructor).
        Side effects: none; no I/O until phase_1().
        Exceptions: none.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.data_url = data_url or _DATA_URL
        self.shard_prefixes = (
            all_shard_prefixes() if shard_prefixes is None else shard_prefixes
        )

    def _run_phase_1(self) -> Path:
        """Download each shard not already on disk.

        Inputs: none.
        Output: self.raw_data_dir (holding archive/ and recent/).
        Side effects: one paced GET per missing shard; writes
            <raw_data_dir>/<tier>/<prefix>.json; shows a tqdm bar; logs
            the counts at INFO.
        Exceptions: requests.RequestException if a GET other than a
            404 still fails after retries.

        Example:
            >>> FabtcgmetaOnlineDecksDownloader().phase_1()
            PosixPath('data/raw/fabtcgmeta_online_decks')
        """
        downloaded = 0
        missing: list[str] = []
        targets = [
            (tier, prefix) for tier in ShardTier for prefix in self.shard_prefixes
        ]

        # Fetch each shard not yet on disk
        for tier, prefix in tqdm(targets, desc="fabtcgmeta shards", unit="shard"):
            destination_path = self.raw_data_dir / tier.value / f"{prefix}.json"
            if destination_path.is_file():
                continue
            self.rate_limiter.wait()
            if self._download_shard(tier, prefix, destination_path):
                downloaded += 1
            else:
                missing.append(f"{tier.value}/{prefix}")

        _logger.info(
            "fabtcgmeta online decks: %d shards downloaded, %d not found %s",
            downloaded,
            len(missing),
            missing[:20],
        )
        return self.raw_data_dir

    def _download_shard(
        self, tier: ShardTier, prefix: str, destination_path: Path
    ) -> bool:
        """GET one shard to destination_path.

        The shard is written beside it as <prefix>.json.part and renamed
        into place once complete: a run killed mid-write (which
        download_to_file's cleanup doesn't catch) leaves only a .part
        file, which the is_file() skip in _run_phase_1 never sees.

        Inputs: tier, prefix, destination_path.
        Output: True if saved, False if the server has no such shard (404).
        Side effects: one GET (retried); writes destination_path.
        Exceptions: requests.RequestException for any failure but a 404.
        """
        partial_path = destination_path.with_name(destination_path.name + ".part")
        try:
            download_to_file(
                f"{self.data_url}/{tier.value}/{prefix}.json", partial_path
            )
        except requests.HTTPError as error:
            if error.response is not None and (
                error.response.status_code == _HTTP_NOT_FOUND
            ):
                return False
            raise
        partial_path.replace(destination_path)
        return True


def all_shard_prefixes() -> tuple[str, ...]:
    """Every 3-hex shard prefix, "000" to "fff".

    Inputs: none. Output: 4,096 lowercase prefixes, in order.
    Side effects: none. Exceptions: none.

    Example:
        >>> all_shard_prefixes()[:2]
        ('000', '001')
    """
    return tuple(
        f"{index:0{_SHARD_PREFIX_HEX_DIGITS}x}"
        for index in range(16**_SHARD_PREFIX_HEX_DIGITS)
    )
