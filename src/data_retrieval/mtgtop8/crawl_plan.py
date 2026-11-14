"""One MTGTop8 format's walk: every deck its search listing named, once,
in a fixed seeded shuffle, so any prefix of the crawl is a uniform
sample of the format.

Phase 1 lands each search page as listing/page-<nnnnn>.jsonl
(ListingRow lines) and merges them into crawl_plan.jsonl, the same line
format in walk order. MtgTop8PlanFile reads the plan as a
resumable_crawl.CrawlPlan with no skipped positions, and also answers
each deck's listing row, so a landed deck carries its event and format.

A format's plan holds at most ~133k rows (Modern, 2026-11-11), ~25 MB,
so it is read whole, unlike Moxfield's 5.7M-line plan.
"""

import hashlib
import random
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator

from src.data_retrieval.mtgtop8.listing import ListingRow
from src.data_retrieval.resumable_crawl import write_text_atomically

PLAN_NAME = "crawl_plan.jsonl"
_PLAN_SEED = 20261111


class MtgTop8PlanFile:
    """crawl_plan.jsonl: one ListingRow line per deck, in walk order."""

    def __init__(self, path: Path) -> None:
        """Inputs: path. Output: none. Side effects: none.
        Exceptions: none."""
        self.path = path

    @cached_property
    def fingerprint(self) -> str:
        """SHA-256 (hex) of the file. Inputs: none. Output: str.
        Side effects: reads it once. Exceptions: OSError.

        Example:
            >>> len(MtgTop8PlanFile(Path("crawl_plan.jsonl")).fingerprint)
            64
        """
        with open(self.path, "rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    @cached_property
    def rows(self) -> tuple[ListingRow, ...]:
        """Every plan row, in walk order. Inputs: none.
        Output: tuple of ListingRow. Side effects: reads the file once.
        Exceptions: OSError; ValueError for a malformed line.

        Example:
            >>> MtgTop8PlanFile(Path("crawl_plan.jsonl")).rows[0].deck_id
            262535
        """
        return tuple(read_listing_rows(self.path))

    @cached_property
    def length(self) -> int:
        """Rows in the plan. Inputs: none. Output: int.
        Side effects: reads the file once (through rows).
        Exceptions: OSError; ValueError for a malformed line.

        Example:
            >>> MtgTop8PlanFile(Path("crawl_plan.jsonl")).length
            81436
        """
        return len(self.rows)

    def ids_from(self, position: int) -> Iterator[tuple[int, int]]:
        """(position, deck id) for every row from position on.

        Inputs: position (>= 0). Output: an iterator over the pairs.
        Side effects: reads the file once (through rows).
        Exceptions: OSError; ValueError for a malformed line.

        Example:
            >>> next(MtgTop8PlanFile(Path("crawl_plan.jsonl")).ids_from(0))
            (0, 262535)
        """
        for walk_position in range(position, self.length):
            yield walk_position, self.rows[walk_position].deck_id

    def row_for(self, deck_id: int) -> ListingRow:
        """The listing row of deck_id.

        Inputs: deck_id (one of the plan's). Output: ListingRow.
        Side effects: builds the id index on first call.
        Exceptions: KeyError if the plan has no such deck.

        Example:
            >>> plan.row_for(262535).format
            <MtgTop8Format.PAUPER: 'PAU'>
        """
        return self._rows_by_deck_id[deck_id]

    @cached_property
    def _rows_by_deck_id(self) -> dict[int, ListingRow]:
        """deck id -> its row. Inputs: none. Output: dict.
        Side effects: reads the file once (through rows).
        Exceptions: OSError; ValueError for a malformed line."""
        return {row.deck_id: row for row in self.rows}


def merge_and_shuffle(landed_paths: Iterable[Path]) -> list[ListingRow]:
    """Every deck in the landed listing pages, once (the first page to
    list it wins: forward paging repeats a deck when newer ones push it
    down a page), sorted by deck id and then shuffled with
    random.Random(_PLAN_SEED), so the plan depends only on the set of
    rows, not on when each page was fetched.

    Inputs: landed_paths (listing pages, in page order).
    Output: the rows in plan order. Side effects: reads the files.
    Exceptions: OSError; ValueError for a malformed line or no rows.

    Example:
        >>> len(merge_and_shuffle(sorted(Path("listing").glob("*.jsonl"))))
        81436
    """
    first_listing: dict[int, ListingRow] = {}

    # Keep each deck's first listing
    for path in landed_paths:
        for row in read_listing_rows(path):
            first_listing.setdefault(row.deck_id, row)
    if not first_listing:
        raise ValueError("the landed MTGTop8 listing pages hold no decks")

    # Fixed order: by id, then the seeded shuffle
    result = [first_listing[deck_id] for deck_id in sorted(first_listing)]
    random.Random(_PLAN_SEED).shuffle(result)
    return result


def read_listing_rows(path: Path) -> list[ListingRow]:
    """The ListingRows of one landed page or plan file.
    Inputs: path. Output: list of ListingRow. Side effects: reads path.
    Exceptions: OSError; ValueError for a malformed line."""
    with open(path, encoding="utf-8", newline="\n") as file:
        return [ListingRow.from_line(line) for line in file if line.strip()]


def write_listing_rows(path: Path, rows: Iterable[ListingRow]) -> None:
    """Write rows, one JSON line each, atomically: a landed page, or the
    plan (rows in walk order). An empty page writes an empty file.
    Inputs: path, rows. Output: none. Side effects: writes path.
    Exceptions: OSError."""
    write_text_atomically(path, "".join(row.to_line() + "\n" for row in rows))
