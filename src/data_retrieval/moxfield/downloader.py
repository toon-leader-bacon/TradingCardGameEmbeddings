"""Downloads Moxfield decks (MTG) one deck at a time through Moxfield's
undocumented read API, for the decks Moxfield's sitemap lists.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Each deck is unwrapped
by pruning (pruning.py); refining (formats, boards, card matching) is
data_refinement's job. The crawl's files, walk loop and HTTP reading
are the shared resumable_crawl/ package's.

TERMS AND PACE (2026-11-05): moxfield.com/robots.txt allows deck pages
and lists the sitemap. api2.moxfield.com/robots.txt is "User-agent: * /
Disallow: /"; the user read it as an oversight and decided to build
this crawler for personal, noncommercial research. Pace is the
project's 60/min, with a plain, honest User-Agent and no attempt to get
past Cloudflare's challenge (a challenge page is a transient failure).
Nothing about deck owners (user names, ids) is kept.

ACCESS (probed 2026-11-05/06): a deck page is a 1.7 KB script shell; the
deck lives at api2.moxfield.com/v3/decks/all/<publicId>. Cards carry
Scryfall printing ids, not oracle ids. There is no bulk route: the
search listing (/v2/decks/search) gives 100 decks a page without card
lists and stops at 10,000 results. Public ids are 22 random base64url
characters, so ids can't be walked like Archidekt's: the sitemap is the
only index. sitemap.xml lists 1,965 sitemap-deck-<n>.xml files of
~2,500-3,350 decks each (plus "/primer" URLs, dropped): ~5.7M decks,
~66 days at 60/min.

WALK ORDER (crawl_plan.py): every sitemap deck once, in a fixed seeded
shuffle. Decks listed after phase 1 are beyond the plan; a later plan
can cover them (each plan row keeps the sitemap's lastmod for that).

    1. phase_1() lands each deck sitemap as sitemaps/deck-<nnnnn>.tsv
       (resumable, one paced GET each, ~35 min), then merges them into
       crawl_plan.tsv. An existing plan is kept; a cursor without a plan
       is refused.
    2. phase_2() walks the plan from cursor.json (resumable_crawl's
       ResumableCrawl), one paced GET per deck, for up to
       max_decks_per_session decks: a deck lands as one pruned row in
       decks-<nnnnn>.jsonl(.gz); a 404 lands nothing; another 4xx (but
       429) or an unreadable deck is a reject; network errors, 5xx, a
       429 beyond its waits and a non-JSON 200 are transient.
       _MAX_CONSECUTIVE_REJECTS rejects in a row stop the session
       (CrawlBlockedError): Moxfield is likely refusing us.
"""

import logging
from pathlib import Path
from typing import ClassVar

from tqdm import tqdm

from src.data_retrieval.downloader import Downloader
from src.data_retrieval.moxfield.crawl_plan import (
    PLAN_NAME,
    MoxfieldPlanFile,
    parse_deck_sitemap_urls,
    merge_and_shuffle,
    parse_sitemap_decks,
    write_sitemap_decks,
)
from src.data_retrieval.moxfield.pruning import prune_deck
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.resumable_crawl import (
    PERMANENT_STATUSES,
    FetchOutcome,
    ResumableCrawl,
    WalkPolicy,
    crawl_get,
    read_outcome,
    refuse_orphan_cursor,
    wrap_as_found,
)

_logger = logging.getLogger(__name__)

_SITE_URL = "https://moxfield.com"
_API_URL = "https://api2.moxfield.com"
_SITEMAPS_DIR_NAME = "sitemaps"
_MAX_CONSECUTIVE_REJECTS = 20


class MoxfieldDeckDownloader(Downloader):
    """Walks the sitemap's Moxfield decks and lands one pruned row each.

    Single-consumer to src/data_refinement/deck_box/moxfield/ (not yet
    built).
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/moxfield")

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        site_url: str | None = None,
        api_url: str | None = None,
        max_decks_per_session: int | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter, raw_data_dir: see Downloader.__init__ (60/min).
            site_url: the site root (sitemaps), no trailing slash.
            api_url: the API root (decks), no trailing slash.
            max_decks_per_session: decks one phase_2() call fetches; None
                walks to the end of the plan.
        Output: none. Side effects: none; no I/O until phase_1().
        Exceptions: ValueError if max_decks_per_session < 1.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.site_url = site_url or _SITE_URL
        self.api_url = api_url or _API_URL
        if max_decks_per_session is not None and max_decks_per_session < 1:
            raise ValueError(
                f"max_decks_per_session must be >= 1, got {max_decks_per_session}"
            )
        self.max_decks_per_session = max_decks_per_session

    def _run_phase_1(self) -> Path:
        """Land every deck sitemap, then write crawl_plan.tsv, unless the
        plan exists.

        Inputs: none. Output: the plan's path.
        Side effects: one paced GET for sitemap.xml and one per deck
            sitemap not yet landed (a tqdm bar); writes sitemaps/*.tsv and
            the plan.
        Exceptions: requests.RequestException after retries; ValueError
            if sitemap.xml lists no deck sitemaps or the merged sitemaps
            no decks; RuntimeError if a cursor exists without a plan.

        Example:
            >>> MoxfieldDeckDownloader().phase_1()
            PosixPath('data/raw/moxfield/crawl_plan.tsv')
        """
        plan_path = self.raw_data_dir / PLAN_NAME
        if plan_path.is_file():
            return plan_path
        refuse_orphan_cursor(self.raw_data_dir, PLAN_NAME)

        # Land each deck sitemap not yet on disk
        sitemaps_dir = self.raw_data_dir / _SITEMAPS_DIR_NAME
        sitemaps_dir.mkdir(parents=True, exist_ok=True)
        index_xml = crawl_get(f"{self.site_url}/sitemap.xml", self.rate_limiter).text
        sitemap_urls = parse_deck_sitemap_urls(index_xml)
        for number, url in tqdm(
            sitemap_urls.items(), desc="Moxfield sitemaps", unit="sitemap"
        ):
            landed_path = _landed_sitemap_path(sitemaps_dir, number)
            if not landed_path.is_file():
                self._land_sitemap(url, landed_path)

        # Merge into the plan: unique ids, latest lastmod, shuffled
        landed_paths = [
            _landed_sitemap_path(sitemaps_dir, number) for number in sitemap_urls
        ]
        landed_paths = [path for path in landed_paths if path.is_file()]
        decks = merge_and_shuffle(landed_paths)
        _logger.info(
            "Moxfield plan: %d decks from %d sitemaps (%d listed, %d skipped)",
            len(decks),
            len(landed_paths),
            len(sitemap_urls),
            len(sitemap_urls) - len(landed_paths),
        )
        write_sitemap_decks(plan_path, decks)
        return plan_path

    def phase_2(self) -> Path:
        """Walk the plan from the cursor, for up to max_decks_per_session
        decks (see resumable_crawl.ResumableCrawl.walk).

        Inputs: none (reads crawl_plan.tsv and cursor.json).
        Output: raw_data_dir.
        Side effects: one paced GET per deck (more on a retry); shard and
            reject rows; cursor.json after each deck; a tqdm bar.
        Exceptions: FileNotFoundError if phase_1() never wrote a plan;
            ValueError if the cursor belongs to another plan, or a 200
            body is not JSON; requests.RequestException for a transient
            failure past its retries; CrawlBlockedError after
            _MAX_CONSECUTIVE_REJECTS rejects in a row.

        Example:
            >>> MoxfieldDeckDownloader(max_decks_per_session=3600).phase_2()
            PosixPath('data/raw/moxfield')
        """
        crawl = ResumableCrawl[str](
            self.raw_data_dir,
            MoxfieldPlanFile(self.raw_data_dir / PLAN_NAME),
            "Moxfield decks",
            WalkPolicy(max_consecutive_rejects=_MAX_CONSECUTIVE_REJECTS),
        )
        crawl.walk(self._fetch_deck, self.max_decks_per_session)
        return self.raw_data_dir

    def _land_sitemap(self, url: str, landed_path: Path) -> None:
        """GET one deck sitemap and write its decks (parse_sitemap_decks) to
        landed_path, atomically (fsynced), so a landed file is always
        whole. A sitemap answering 404 (listed, then removed) is logged
        and skipped, not landed: phase 1 can still finish.
        Inputs: url, landed_path. Output: none.
        Side effects: one paced GET (crawl_get); writes landed_path.
        Exceptions: requests.RequestException (another error status, or
            transient); OSError."""
        response = crawl_get(url, self.rate_limiter, frozenset({404}))
        if response.status_code == 404:
            _logger.warning("Moxfield deck sitemap %s answered 404; skipped", url)
            return
        write_sitemap_decks(landed_path, parse_sitemap_decks(response.text))

    def _fetch_deck(self, public_id: str) -> FetchOutcome:
        """GET /v3/decks/all/<public_id> (crawl_get, tolerating
        PERMANENT_STATUSES), read by read_outcome with prune_deck.
        Inputs: public_id. Output: FetchOutcome.
        Side effects: one paced GET (more on a retry).
        Exceptions: requests.RequestException (transient); ValueError for
            a 2xx whose body is not JSON (transient)."""
        response = crawl_get(
            f"{self.api_url}/v3/decks/all/{public_id}",
            self.rate_limiter,
            PERMANENT_STATUSES,
        )
        return read_outcome(response, wrap_as_found(prune_deck), f"deck {public_id}")


def _landed_sitemap_path(sitemaps_dir: Path, number: int) -> Path:
    """Where deck sitemap number lands. Inputs: sitemaps_dir, number.
    Output: <sitemaps_dir>/deck-<nnnnn>.tsv. Side effects: none."""
    return sitemaps_dir / f"deck-{number:05d}.tsv"
