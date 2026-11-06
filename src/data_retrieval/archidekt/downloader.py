"""Downloads Archidekt decks (MTG, mostly Commander) one deck id at a
time through Archidekt's open, undocumented read API.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Each deck is unwrapped
by pruning (pruning.py); refining (formats, boards, card matching) is
data_refinement's job. The crawl's files, walk loop and HTTP reading
are the shared resumable_crawl/ package's.

TERMS AND PACE: robots.txt disallows only login, compare, playtester
and sandbox pages. The ToS grants access for personal, noncommercial
use, which the user ruled this research is (2026-11-05). Staff on the
forum: "You're welcome to use our API as is" (thread 11704454).
Evidence of a server throttle at about 80 requests/min (staff, thread
9724958); this downloader keeps the project's 60/min to be polite.

ACCESS (probed 2026-11-05): there is no bulk route. /api/decks/v3/
lists 60 decks a page without cards, capped at 1,000 results, whatever
pageSize says; /api/decks/<id>/small/ has no cards. Only
/api/decks/<id>/ carries cards, so the crawl walks deck ids. Ids are
dense: of 40 random ids below the newest (~28.1M), 19 were decks and
21 answered 404 {"error":"Deck not found."} (deleted or private). So
~13M decks, and the whole range at 60/min is ~325 days: any prefix of
the walk has to be useful on its own.

WALK ORDER (the user's choice; crawl_plan.py): first the ~17,500 decks
the sitemap lists (sitemaps/decks.xml; popular, recently active, listed
Commander decks; a tiny slice, not an index), then every id in
[1, newest] in a fixed spread order. Decks created after phase 1 are
beyond the plan; a later plan can cover them.

UNLISTED DECKS (the user's choice): an owner can mark a deck unlisted
(hidden from Archidekt's search, readable by link). They are kept with
their "unlisted" flag, so refinement can leave them out of training.
Nothing about the owner is kept.

    1. phase_1() fetches the sitemap's deck ids and the newest deck id
       (the list ordered by -createdAt) and writes crawl_plan.json. An
       existing plan is kept, so the walk never changes under a running
       crawl; with a cursor but no plan, it refuses (the cursor belongs
       to a lost plan).
    2. phase_2() walks the plan from cursor.json (resumable_crawl's
       ResumableCrawl), one paced GET per id, for up to
       max_decks_per_session ids: a deck lands as one pruned row in
       decks-<nnnnn>.jsonl(.gz); a 404 lands nothing; another 4xx (but
       429) or an unreadable deck is a reject; network errors, 5xx, a
       429 beyond its waits and a non-JSON 200 are transient (the next
       session retries the id, up to 3 sessions).
"""

import logging
import re
from pathlib import Path
from typing import ClassVar

from src.data_retrieval.archidekt.crawl_plan import ARCHIDEKT_ITEMS, PLAN_NAME
from src.data_retrieval.archidekt.pruning import prune_deck
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.resumable_crawl import (
    PERMANENT_STATUSES,
    FetchOutcome,
    ResumableCrawl,
    SitemapThenSpreadPlan,
    SpreadOrder,
    crawl_get,
    read_outcome,
    refuse_orphan_cursor,
    wrap_as_found,
    write_text_atomically,
)

_logger = logging.getLogger(__name__)

_SITE_URL = "https://archidekt.com"
_SITEMAP_DECK_PATTERN = re.compile(r"<loc>\s*[^<]*?/decks/(\d+)/?\s*</loc>")


class ArchidektDeckDownloader(Downloader):
    """Walks Archidekt deck ids and lands one pruned row per deck.

    Single-consumer to src/data_refinement/deck_box/archidekt/ (not yet
    built).
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/archidekt")

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        site_url: str | None = None,
        max_decks_per_session: int | None = None,
    ) -> None:
        """
        Inputs:
            rate_limiter: see Downloader.__init__ (default 60/min).
            raw_data_dir: see Downloader.__init__.
            site_url: the site root, no trailing slash. Defaults to
                https://archidekt.com.
            max_decks_per_session: deck ids one phase_2() call fetches
                before returning (retries add GETs beyond this; skipped
                sitemap slots in the spread part don't count). None
                walks to the end of the plan.
        Output: none (constructor).
        Side effects: none; no I/O until phase_1().
        Exceptions: ValueError if max_decks_per_session < 1.
        """
        super().__init__(rate_limiter, raw_data_dir)
        self.site_url = site_url or _SITE_URL
        if max_decks_per_session is not None and max_decks_per_session < 1:
            raise ValueError(
                f"max_decks_per_session must be >= 1, got {max_decks_per_session}"
            )
        self.max_decks_per_session = max_decks_per_session

    def _run_phase_1(self) -> Path:
        """Write crawl_plan.json unless it exists.

        Inputs: none.
        Output: the plan's path.
        Side effects: two paced GETs (the sitemap, the newest-decks
            list) when no plan exists; writes the plan.
        Exceptions: requests.RequestException after retries; ValueError
            if the sitemap lists no decks or the list has no results;
            RuntimeError if a cursor exists without a plan.

        Example:
            >>> ArchidektDeckDownloader().phase_1()
            PosixPath('data/raw/archidekt/crawl_plan.json')
        """
        plan_path = self.raw_data_dir / PLAN_NAME
        if plan_path.is_file():
            return plan_path
        refuse_orphan_cursor(self.raw_data_dir, PLAN_NAME)

        # Gather the targets
        sitemap_ids = self._sitemap_deck_ids()
        newest_id = self._newest_deck_id()
        plan = SitemapThenSpreadPlan(
            tuple(sitemap_ids), SpreadOrder.for_range(newest_id), ARCHIDEKT_ITEMS
        )
        _logger.info(
            "Archidekt plan: %d sitemap decks, then ids 1..%d (stride %d)",
            len(sitemap_ids),
            newest_id,
            plan.spread.stride,
        )

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomically(plan_path, plan.to_json())
        return plan_path

    def phase_2(self) -> Path:
        """Walk the plan from the cursor, for up to max_decks_per_session
        ids (see resumable_crawl.ResumableCrawl.walk).

        Inputs: none (reads crawl_plan.json and cursor.json).
        Output: raw_data_dir.
        Side effects: one paced GET per walk position (more on a
            retry); deck rows to the shards, permanent failures to
            rejects.jsonl; cursor.json after each id; a tqdm bar; logs
            the session's counts.
        Exceptions: FileNotFoundError if phase_1() never wrote a plan;
            ValueError if the cursor belongs to another plan, or a 200
            body is not JSON; requests.RequestException for a transient
            failure that outlasted its retries (see ResumableCrawl.walk
            for where the cursor is left).

        Example:
            >>> ArchidektDeckDownloader(max_decks_per_session=3600).phase_2()
            PosixPath('data/raw/archidekt')
        """
        plan = SitemapThenSpreadPlan.from_json(
            (self.raw_data_dir / PLAN_NAME).read_text(encoding="utf-8"), ARCHIDEKT_ITEMS
        )
        crawl = ResumableCrawl[int](self.raw_data_dir, plan, "Archidekt decks")
        crawl.walk(self._fetch_deck, self.max_decks_per_session)
        return self.raw_data_dir

    def _sitemap_deck_ids(self) -> list[int]:
        """The deck ids sitemaps/decks.xml lists, in file order, without
        repeats.
        Inputs: none. Output: list[int]. Side effects: one paced GET.
        Exceptions: requests.RequestException; ValueError if none."""
        text = crawl_get(f"{self.site_url}/sitemaps/decks.xml", self.rate_limiter).text
        deck_ids = list(
            dict.fromkeys(int(match) for match in _SITEMAP_DECK_PATTERN.findall(text))
        )
        if not deck_ids:
            raise ValueError("Archidekt's deck sitemap lists no decks")
        return deck_ids

    def _newest_deck_id(self) -> int:
        """The highest id on the first page of /api/decks/v3/ ordered by
        -createdAt.
        Inputs: none. Output: int. Side effects: one paced GET.
        Exceptions: requests.RequestException; ValueError if no results
            or the body is not JSON."""
        listing = crawl_get(
            f"{self.site_url}/api/decks/v3/?orderBy=-createdAt", self.rate_limiter
        ).json()
        deck_ids = [int(deck["id"]) for deck in listing.get("results", [])]
        if not deck_ids:
            raise ValueError("Archidekt's newest-decks list has no results")
        return max(deck_ids)

    def _fetch_deck(self, deck_id: int) -> FetchOutcome:
        """GET /api/decks/<id>/ (tolerating PERMANENT_STATUSES), read by
        read_outcome with prune_deck.
        Inputs: deck_id. Output: FetchOutcome.
        Side effects: one paced GET (more on a retry).
        Exceptions: requests.RequestException (transient); ValueError for
            a 200 whose body is not JSON (transient)."""
        response = crawl_get(
            f"{self.site_url}/api/decks/{deck_id}/",
            self.rate_limiter,
            PERMANENT_STATUSES,
        )
        return read_outcome(response, wrap_as_found(prune_deck), f"deck {deck_id}")
