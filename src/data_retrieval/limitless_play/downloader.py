"""Downloads Pokemon TCG tournament standings, with decklists, from the
Limitless play API (play.limitlesstcg.com/api), in two phases.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Each API response is
already JSON, so nothing here is unwrapped beyond tagging each
standings payload with the tournament id it came from.

Limitless runs online grassroots tournaments, mostly on Pokemon TCG
Live, where each player submits a decklist. The API is public and needs
no key (docs.limitlesstcg.com/developer). CONFIRMED LIVE 2026-10-28:

    1. phase_1() pages GET /tournaments?game=PTCG&limit=500&page=N
       (1-based) until an empty page, writing every tournament summary
       ({"id", "name", "date", "format", "players", "organizerId",
       "game"}) to tournaments.jsonl. About 17,000 tournaments, 34 pages.
    2. phase_2() fetches GET /tournaments/{id}/standings for every
       tournament id not yet in standings_manifest.txt, appending
       {"tournament_id": id, "standings": [...]} to standings.jsonl.
       Each standings row carries "decklist": {"pokemon", "trainer",
       "energy": [{"count", "set", "number", "name"}]} (or null), plus
       "placing", "record", "deck" (the archetype), "player", "name",
       "country".

RATE LIMIT: without a key the server allows 50 requests per 5 minutes
(its RateLimit-Policy header: "50-in-5min"; q=50; w=300). This
source's default pacing is therefore 9 requests/min, below that, not
the container's usual 60. Phase 2 takes about 30 hours. A 429 is still
handled: the downloader waits out the window the RateLimit header
reports and retries the same request.

TODO: Check in on the status of the API key application:
https://play.limitlesstcg.com/account/settings/api
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import ClassVar

import requests
from tqdm import tqdm

from src.data_retrieval.download_utils import (
    append_with_manifest,
    call_with_retries,
    read_manifest,
)
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 60
_MAX_FETCH_ATTEMPTS = 3  # network errors: 1 initial try + up to 2 retries
_MAX_RATE_LIMITED_WAITS = 5  # 429s tolerated for one request before failing
_DEFAULT_RATE_LIMITED_WAIT_SECONDS = 300  # the server's window, if unstated
_REQUESTS_PER_MINUTE = 9  # server allows 50 per 5 min; see module docstring
_PAGE_SIZE = 500
_GAME_ID = "PTCG"
# The RateLimit header's seconds until the window resets: '"50-in-5min"; r=0; t=212'
_RATE_LIMIT_RESET_PATTERN = re.compile(r"\bt=(\d+)")


class LimitlessPlayDownloader(Downloader):
    """Downloads Pokemon TCG tournament summaries and standings
    (decklists included) from the Limitless play API.

    Single-consumer to src/data_retrieval/limitless_play/.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/limitless_play")
    DEFAULT_API_URL: ClassVar[str] = "https://play.limitlesstcg.com/api"
    STANDINGS_FILENAME: ClassVar[str] = "standings.jsonl"

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        api_url: str | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__. Defaults to
                RateLimiter(_REQUESTS_PER_MINUTE), not the base class's
                60/min, because of the server's own limit.
            raw_data_dir: see Downloader.__init__.
            api_url: the API's base URL. Defaults to DEFAULT_API_URL.
        Output: none (constructor).
        Side effects: none; no I/O until phase_1()/phase_2().
        Exceptions: none.
        """
        super().__init__(
            rate_limiter or RateLimiter(requests_per_minute=_REQUESTS_PER_MINUTE),
            raw_data_dir,
        )
        self.api_url = api_url or self.DEFAULT_API_URL

    @property
    def tournaments_path(self) -> Path:
        """raw_data_dir/tournaments.jsonl, phase_1()'s output."""
        return self.raw_data_dir / "tournaments.jsonl"

    @property
    def standings_path(self) -> Path:
        """raw_data_dir/standings.jsonl, phase_2()'s output."""
        return self.raw_data_dir / self.STANDINGS_FILENAME

    def _run_phase_1(self) -> Path:
        """Page the PTCG tournament list and write every tournament
        summary to tournaments.jsonl.

        Inputs: none (uses self.api_url, self.raw_data_dir).
        Output: self.tournaments_path.
        Side effects: one paced request per page; creates raw_data_dir;
            overwrites tournaments.jsonl; shows a tqdm bar.
        Exceptions: requests.RequestException if a page still fails
            after retries (a page walk can't skip a page); ValueError
            if no tournaments were found at all.

        Example:
            >>> LimitlessPlayDownloader().phase_1()
        """
        tournaments: list[dict] = []

        # Page until the API returns an empty page
        page = 1
        with tqdm(desc="Limitless tournament pages", unit="page") as progress:
            while True:
                page_tournaments = self._fetch_tournament_page(page)
                if not page_tournaments:
                    break
                tournaments.extend(page_tournaments)
                progress.update(1)
                page += 1

        if not tournaments:
            raise ValueError(f"no {_GAME_ID} tournaments listed at {self.api_url}")

        # Dedup by id, keeping the first sighting and walk order (newest
        # first): a tournament created mid-walk shifts later pages by one
        unique_tournaments: dict[str, dict] = {}
        for tournament in tournaments:
            unique_tournaments.setdefault(tournament["id"], tournament)
        self._write_tournaments(list(unique_tournaments.values()))
        return self.tournaments_path

    def phase_2(self) -> Path:
        """Fetch standings for every listed tournament not already in
        standings_manifest.txt, appending each to standings.jsonl.

        Best-effort per tournament: one that still fails after retries
        is logged and skipped, so a later run retries it. Tournaments
        listing zero players are skipped without a request.

        Inputs: none (reads tournaments.jsonl from phase_1()).
        Output: self.standings_path.
        Side effects: one paced request per new tournament; appends to
            standings.jsonl and standings_manifest.txt (data row first,
            see append_with_manifest); shows a tqdm bar; logs a warning
            per skipped tournament.
        Exceptions: FileNotFoundError if tournaments.jsonl is missing.

        Example:
            >>> downloader = LimitlessPlayDownloader()
            >>> downloader.phase_1()
            >>> downloader.phase_2()
        """
        tournament_ids = self._read_tournament_ids_with_players()
        manifest_path = self.raw_data_dir / "standings_manifest.txt"
        already_downloaded = read_manifest(manifest_path)

        # Fetch each new tournament's standings
        for tournament_id in tqdm(
            tournament_ids, desc="Limitless standings", unit="tournament"
        ):
            if tournament_id in already_downloaded:
                continue
            try:
                standings = self._fetch_standings(tournament_id)
            except requests.RequestException as error:
                _logger.warning(
                    "Limitless standings %s failed: %s", tournament_id, error
                )
                continue
            append_with_manifest(
                self.standings_path,
                manifest_path,
                json.dumps({"tournament_id": tournament_id, "standings": standings}),
                tournament_id,
            )

        return self.standings_path

    def _fetch_tournament_page(self, page: int) -> list[dict]:
        """One page of the PTCG tournament list (1-based page).

        Inputs: page (int). Output: the page's tournament summaries.
        Side effects: paced request(s), see _get_json.
        Exceptions: as _get_json.
        """
        return self._get_json(
            f"{self.api_url}/tournaments",
            {"game": _GAME_ID, "limit": _PAGE_SIZE, "page": page},
        )

    def _fetch_standings(self, tournament_id: str) -> list[dict]:
        """One tournament's standings rows.

        Inputs: tournament_id (str). Output: the standings rows.
        Side effects: paced request(s), see _get_json.
        Exceptions: as _get_json.
        """
        return self._get_json(
            f"{self.api_url}/tournaments/{tournament_id}/standings", {}
        )

    def _get_json(self, url: str, params: dict[str, str | int]) -> list[dict]:
        """GET url and decode its JSON list body, paced, retrying
        network errors via call_with_retries and waiting out 429s.

        The two budgets are separate: each request (up to
        _MAX_RATE_LIMITED_WAITS + 1 of them) gets _MAX_FETCH_ATTEMPTS
        tries against network errors and non-429 error statuses, and a
        429 response is waited out (_rate_limited_wait_seconds) and
        requested again, at most _MAX_RATE_LIMITED_WAITS times.

        Inputs: url, params (query parameters).
        Output: the decoded JSON list.
        Side effects: one or more paced requests; may sleep minutes.
        Exceptions: requests.RequestException once a request's retries
            are exhausted, or requests.HTTPError when it is still
            rate limited after every wait; ValueError if the body isn't
            JSON.
        """

        def _attempt() -> requests.Response:
            self.rate_limiter.wait()
            response = requests.get(
                url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS
            )
            # A 429 is returned, not raised: call_with_retries would
            # retry it after 2 s, inside the server's window
            if response.status_code != 429:
                response.raise_for_status()
            return response

        # Request, waiting out each 429 except the last
        for waits_so_far in range(_MAX_RATE_LIMITED_WAITS + 1):
            response = call_with_retries(_attempt, max_attempts=_MAX_FETCH_ATTEMPTS)
            if response.status_code != 429:
                return response.json()
            if waits_so_far == _MAX_RATE_LIMITED_WAITS:
                break
            wait_seconds = self._rate_limited_wait_seconds(response)
            _logger.warning("Limitless rate limited; waiting %.0f s", wait_seconds)
            time.sleep(wait_seconds)

        raise requests.HTTPError(
            f"still rate limited after {_MAX_RATE_LIMITED_WAITS} waits: {url}"
        )

    def _rate_limited_wait_seconds(self, response: requests.Response) -> float:
        """Seconds to wait after a 429: the Retry-After header, else the
        RateLimit header's t=<seconds>, else
        _DEFAULT_RATE_LIMITED_WAIT_SECONDS.

        Inputs: response (a 429). Output: seconds (float).
        Side effects: none. Exceptions: none.
        """
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return float(retry_after)
        reset_match = _RATE_LIMIT_RESET_PATTERN.search(
            response.headers.get("RateLimit", "")
        )
        if reset_match is not None:
            # One second past the reset, so the next request lands after it
            return float(reset_match.group(1)) + 1
        return float(_DEFAULT_RATE_LIMITED_WAIT_SECONDS)

    def _write_tournaments(self, tournaments: list[dict]) -> None:
        """Overwrite tournaments.jsonl with one summary per line.

        Inputs: tournaments. Output: none.
        Side effects: creates raw_data_dir; writes tournaments.jsonl.
        Exceptions: OSError on a write failure.
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.tournaments_path.write_text(
            "".join(json.dumps(tournament) + "\n" for tournament in tournaments),
            encoding="utf-8",
        )

    def _read_tournament_ids_with_players(self) -> list[str]:
        """Ids of tournaments.jsonl's tournaments with players > 0, in
        file order (newest first).

        Inputs: none. Output: tournament ids.
        Side effects: reads tournaments.jsonl.
        Exceptions: FileNotFoundError if phase_1() hasn't run.
        """
        result: list[str] = []
        with open(self.tournaments_path, encoding="utf-8") as tournaments_file:
            for line in tournaments_file:
                if not line.strip():
                    continue
                tournament = json.loads(line)
                if tournament.get("players"):
                    result.append(tournament["id"])
        return result
