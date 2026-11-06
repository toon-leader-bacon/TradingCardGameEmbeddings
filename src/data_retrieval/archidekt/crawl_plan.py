"""Archidekt's walk: the sitemap's deck ids first, then every id in
[1, newest] in a fixed spread order, so each prefix of the walk is a
near-uniform sample over 2019 to today. A
resumable_crawl.SitemapThenSpreadPlan of int deck ids, persisted as
crawl_plan.json ({"sitemap_ids": [...], "spread": {...}}).
"""

from src.data_retrieval.download_utils import require_int
from src.data_retrieval.resumable_crawl import PlanItems

PLAN_NAME = "crawl_plan.json"

# Items are deck ids: an item is its own id
ARCHIDEKT_ITEMS = PlanItems[int](
    id_of_item=int,
    item_of_id=int,
    read_json_item=require_int,
    sitemap_key="sitemap_ids",
)
