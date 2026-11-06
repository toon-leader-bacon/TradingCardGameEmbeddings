"""A resumable, one-item-at-a-time crawl, shared by the long deck crawls
(archidekt/, moxfield/, deckstats/):

  - crawl_files.py: the on-disk state (JsonlShards, RejectLog,
    CursorFile) and the durable append helpers;
  - fetch.py: crawl_get and read_outcome, one GET read into a
    FetchOutcome (ItemFound | ItemNotFound | ItemRejected);
  - walk.py: ResumableCrawl.walk(), one session over a CrawlPlan;
  - spread_order.py and sitemap_spread_plan.py: SpreadOrder and
    SitemapThenSpreadPlan, the "sitemap first, then every id" plan
    (archidekt/, deckstats/; each crawler supplies its PlanItems).

Which ids, in which order, is each crawler's own plan: the walk sees it
only through the CrawlPlan Protocol (fingerprint, length, ids_from).
"""

from src.data_retrieval.resumable_crawl.crawl_files import (
    CURSOR_NAME,
    REJECTS_NAME,
    SHARD_ROWS,
    Cursor,
    CursorFile,
    JsonlShards,
    RejectLog,
    append_line,
    cut_torn_last_line,
    gzip_and_remove,
    refuse_orphan_cursor,
    write_text_atomically,
)
from src.data_retrieval.resumable_crawl.fetch import (
    PERMANENT_STATUSES,
    REDIRECT_STATUSES,
    FetchOutcome,
    ItemFound,
    ItemNotFound,
    ItemRejected,
    crawl_get,
    read_outcome,
    wrap_as_found,
)
from src.data_retrieval.resumable_crawl.sitemap_spread_plan import (
    PlanItems,
    SitemapThenSpreadPlan,
)
from src.data_retrieval.resumable_crawl.spread_order import SpreadOrder
from src.data_retrieval.resumable_crawl.walk import (
    CrawlBlockedError,
    CrawlPlan,
    ResumableCrawl,
    SessionCounts,
    WalkPolicy,
)

__all__ = [
    "CURSOR_NAME",
    "PERMANENT_STATUSES",
    "REDIRECT_STATUSES",
    "REJECTS_NAME",
    "SHARD_ROWS",
    "CrawlBlockedError",
    "CrawlPlan",
    "Cursor",
    "CursorFile",
    "FetchOutcome",
    "ItemFound",
    "ItemNotFound",
    "ItemRejected",
    "PlanItems",
    "JsonlShards",
    "RejectLog",
    "ResumableCrawl",
    "SessionCounts",
    "SitemapThenSpreadPlan",
    "SpreadOrder",
    "WalkPolicy",
    "append_line",
    "crawl_get",
    "cut_torn_last_line",
    "read_outcome",
    "gzip_and_remove",
    "refuse_orphan_cursor",
    "wrap_as_found",
    "write_text_atomically",
]
