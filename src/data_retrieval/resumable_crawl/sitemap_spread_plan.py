"""The "sitemap first, then every id" plan two crawls share (archidekt/,
deckstats/): a sitemap's items, then every id in [1, n] in a fixed
SpreadOrder, skipping ids the sitemap holds. Persisted as one JSON
file; a CrawlPlan.

A position counts every sitemap item and every spread slot (a skipped
one too), so positions never shift and one integer is the cursor.
What an item is (an Archidekt deck id, a Deckstats "<id>-<slug>" path)
is the crawler's PlanItems.
"""

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Generic, Iterator, TypeVar

from src.data_retrieval.resumable_crawl.spread_order import SpreadOrder

Item = TypeVar("Item", int, str)


@dataclass(frozen=True)
class PlanItems(Generic[Item]):
    """How one crawler's plan items behave: their id, the item for a
    spread id, their JSON check, and their JSON key."""

    id_of_item: Callable[[Item], int]  # a sitemap item's deck id
    item_of_id: Callable[[int], Item]  # a spread id as an item
    read_json_item: Callable[[Any], Item]  # a parsed JSON value, checked
    sitemap_key: str  # the plan JSON's key for the sitemap items


@dataclass(frozen=True)
class SitemapThenSpreadPlan(Generic[Item]):
    """The walk: sitemap_items first, then spread's slots whose id is not
    a sitemap item's.

    Invariants (checked in __post_init__): every sitemap item's id is
    positive, and no id repeats. A sitemap id above spread.n (a deck
    created between the plan's two lookups) is harmless: it is walked
    once, in the sitemap part.
    """

    sitemap_items: tuple[Item, ...]
    spread: SpreadOrder
    items: PlanItems[Item] = field(repr=False, compare=False)
    _sitemap_ids: frozenset[int] = field(
        init=False, repr=False, compare=False, default=frozenset()
    )

    def __post_init__(self) -> None:
        """Inputs: the fields. Output: none. Side effects: none.
        Exceptions: ValueError if an invariant fails (or an item has no
        id: id_of_item's ValueError)."""
        ids = [self.items.id_of_item(item) for item in self.sitemap_items]
        id_set = frozenset(ids)
        if len(id_set) != len(ids):
            raise ValueError("plan sitemap ids repeat")
        if any(item_id < 1 for item_id in ids):
            raise ValueError("plan sitemap ids must be positive")
        # Frozen: the cached set is set once, here
        object.__setattr__(self, "_sitemap_ids", id_set)

    @classmethod
    def over_sitemap(
        cls, sitemap_items: tuple[Item, ...], items: PlanItems[Item]
    ) -> "SitemapThenSpreadPlan[Item]":
        """The plan for these sitemap items, spread over [1, their
        highest id] (for a site whose sitemap reaches its newest decks).

        Inputs: sitemap_items (non-empty), items.
        Output: SitemapThenSpreadPlan. Side effects: none.
        Exceptions: ValueError if sitemap_items is empty or breaks an
            invariant.

        Example:
            >>> SitemapThenSpreadPlan.over_sitemap((7,), ARCHIDEKT_ITEMS).length
            8
        """
        if not sitemap_items:
            raise ValueError("a plan needs at least one sitemap item")
        newest_id = max(items.id_of_item(item) for item in sitemap_items)
        return cls(sitemap_items, SpreadOrder.for_range(newest_id), items)

    @property
    def length(self) -> int:
        """Total walk positions: len(sitemap_items) + spread.n.
        Inputs: none. Output: int. Side effects: none. Exceptions: none.

        Example:
            >>> SitemapThenSpreadPlan((7,), SpreadOrder(3, 1, 0), ARCHIDEKT_ITEMS).length
            4
        """
        return len(self.sitemap_items) + self.spread.n

    def ids_from(self, position: int) -> Iterator[tuple[int, Item]]:
        """(walk position, item) from position to the end, skipping spread
        slots that hold a sitemap id (checked against a set built once per
        plan, not per call).

        Inputs: position (>= 0). Output: iterator of (position, item).
        Side effects: none. Exceptions: none.

        Example:
            >>> plan = SitemapThenSpreadPlan((7, 2), SpreadOrder(3, 1, 0),
            ...                              ARCHIDEKT_ITEMS)
            >>> list(plan.ids_from(0))
            [(0, 7), (1, 2), (2, 1), (4, 3)]
        """
        sitemap_count = len(self.sitemap_items)
        for walk_position in range(position, self.length):
            if walk_position < sitemap_count:
                yield walk_position, self.sitemap_items[walk_position]
                continue
            spread_id = self.spread.id_at(walk_position - sitemap_count)
            if spread_id not in self._sitemap_ids:
                yield walk_position, self.items.item_of_id(spread_id)

    @cached_property
    def fingerprint(self) -> str:
        """The SHA-256 of to_json() (hex), stored with the cursor so a
        cursor never walks another plan. Cached: the cursor is written
        once a second. Inputs: none. Output: str. Side effects: none.

        Example:
            >>> len(plan.fingerprint)
            64
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """The plan file's text: {<sitemap_key>: [...], "spread": {"n",
        "stride", "offset"}}, one line.
        Inputs: none. Output: str. Side effects: none. Exceptions: none.

        Example:
            >>> SitemapThenSpreadPlan((7,), SpreadOrder(3, 1, 0),
            ...                       ARCHIDEKT_ITEMS).to_json()
            '{"sitemap_ids": [7], "spread": {"n": 3, "stride": 1, "offset": 0}}'
        """
        return json.dumps(
            {
                self.items.sitemap_key: list(self.sitemap_items),
                "spread": self.spread.to_dict(),
            }
        )

    @classmethod
    def from_json(
        cls, text: str, items: PlanItems[Item]
    ) -> "SitemapThenSpreadPlan[Item]":
        """Parse a plan file's text.

        Inputs: text, items (the crawler's PlanItems). Output: the plan.
        Side effects: none.
        Exceptions: ValueError for a malformed plan or a broken invariant.

        Example:
            >>> SitemapThenSpreadPlan.from_json(plan.to_json(), ARCHIDEKT_ITEMS) == plan
            True
        """
        try:
            raw = json.loads(text)
            sitemap_items = tuple(
                items.read_json_item(item) for item in raw[items.sitemap_key]
            )
            spread = SpreadOrder.from_dict(raw["spread"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed plan: {error!r}") from error
        return cls(sitemap_items, spread, items)
