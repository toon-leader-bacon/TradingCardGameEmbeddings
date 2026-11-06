"""The walk loop of a resumable crawl: ResumableCrawl.walk() runs one
session over a CrawlPlan from its cursor.

Per id, the caller's fetch answers one FetchOutcome variant (ItemFound:
one shard row; ItemNotFound: nothing; ItemRejected: one reject row),
then the cursor moves past it. A transient failure (fetch raising
requests.RequestException or ValueError) is raised out of the walk with
the cursor still on that id and its failure count raised; after
WalkPolicy.max_session_failures sessions fail on one id, it is rejected
("transient x N") and the cursor moves on, so one broken id can't stall
the crawl. With WalkPolicy.max_consecutive_rejects set, that many
rejects in a row raise CrawlBlockedError: a site refusing every request
answers 403 to each, which would otherwise burn through the plan as
rejects (the ids stay in rejects.jsonl for a later re-walk).

SessionCounts counts what a session's fetches answered: an id rejected
for failing max_session_failures sessions raises out of the walk, so
it counts in neither SessionCounts.rejected nor the run of consecutive
rejects (it is in rejects.jsonl).

The cursor is replaced after each id, after its row, so a crash repeats
at most one id (a duplicate row; deck identity downstream is the card
multiset).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, Iterator, Protocol, TypeVar

import requests
from tqdm import tqdm

from src.data_retrieval.resumable_crawl.crawl_files import (
    CURSOR_NAME,
    REJECTS_NAME,
    SHARD_ROWS,
    CursorFile,
    JsonlShards,
    RejectLog,
)
from src.data_retrieval.resumable_crawl.fetch import (
    FetchOutcome,
    ItemFound,
    ItemNotFound,
    ItemRejected,
)

_logger = logging.getLogger(__name__)

ItemId = TypeVar("ItemId", int, str)
ItemIdCo = TypeVar("ItemIdCo", int, str, covariant=True)


class CrawlPlan(Protocol[ItemIdCo]):
    """What a walk needs of a plan: SitemapThenSpreadPlan (Archidekt,
    Deckstats) and Moxfield's MoxfieldPlanFile both have it."""

    @property
    def fingerprint(self) -> str:
        """The plan's SHA-256 (hex): a cursor belongs to one plan.
        Inputs: none. Output: 64 hex characters. Side effects: none
        beyond the plan's own (a file plan reads its file once).

        Example:
            >>> plan.fingerprint
            '68ec662f...'
        """
        ...

    @property
    def length(self) -> int:
        """Walk positions in the plan (the cursor's end value), skipped
        ones included. Inputs: none. Output: int >= 0.

        Example:
            >>> plan.length
            28144042
        """
        ...

    def ids_from(self, position: int) -> Iterator[tuple[int, ItemIdCo]]:
        """(position, id) pairs from position on, in walk order; a plan
        may skip positions (yield none for them).
        Inputs: position (0 <= position <= length). Output: an iterator.

        Example:
            >>> next(plan.ids_from(17520))
            (17520, 1)
        """
        ...


@dataclass
class SessionCounts:
    """One walk()'s outcomes, one count per FetchOutcome variant."""

    found: int = 0
    not_found: int = 0
    rejected: int = 0


@dataclass(frozen=True)
class WalkPolicy:
    """When a walk gives up on an id, or on the session."""

    max_session_failures: int = 3  # sessions failing on one id before it's rejected
    max_consecutive_rejects: int | None = None  # None: never stop on rejects

    def __post_init__(self) -> None:
        """Inputs: the fields. Output: none. Side effects: none.
        Exceptions: ValueError if max_session_failures < 1 or
        max_consecutive_rejects < 1."""
        if self.max_session_failures < 1:
            raise ValueError(
                f"max_session_failures must be >= 1, got {self.max_session_failures}"
            )
        if (
            self.max_consecutive_rejects is not None
            and self.max_consecutive_rejects < 1
        ):
            raise ValueError(
                "max_consecutive_rejects must be >= 1 or None, got "
                f"{self.max_consecutive_rejects}"
            )


class CrawlBlockedError(RuntimeError):
    """max_consecutive_rejects rejects in a row: the site is likely
    refusing this client, not answering for single ids."""


@dataclass(frozen=True)
class ResumableCrawl(Generic[ItemId]):
    """One crawl directory's state (shards, rejects, cursor) for one plan,
    and the walk over them."""

    directory: Path
    plan: CrawlPlan[ItemId]
    label: str  # names the crawl in logs and the tqdm bar ("Moxfield decks")
    policy: WalkPolicy = field(default_factory=WalkPolicy)
    rows_per_shard: int = SHARD_ROWS

    def walk(
        self, fetch: Callable[[ItemId], FetchOutcome], max_items: int | None = None
    ) -> SessionCounts:
        """Fetch and land the plan's ids from the cursor on, at most
        max_items of them (None: to the plan's end), advancing the cursor
        after each. If only skipped positions remain after the last id,
        the cursor moves to plan.length: the plan is done.

        Inputs: fetch (one paced GET, classified, usually through
            read_outcome; raises requests.RequestException or ValueError
            for a transient failure), max_items (>= 1 or None).
        Output: the session's counts (also logged).
        Side effects: appends shard and reject rows; replaces cursor.json
            after every id; a tqdm bar.
        Exceptions: ValueError if max_items < 1 or the cursor belongs to
            another plan; the transient error from fetch (cursor on that
            id, failures raised; or, at max_session_failures, the id
            rejected and the cursor past it); CrawlBlockedError after
            max_consecutive_rejects rejects in a row (cursor past them);
            OSError.

        Example:
            >>> crawl = ResumableCrawl(Path("data/raw/x"), plan, "X decks")
            >>> crawl.walk(lambda item_id: ItemNotFound(), max_items=2)
            SessionCounts(found=0, not_found=2, rejected=0)
        """
        result = SessionCounts()

        # Validate inputs, and find where this session starts and stops
        if max_items is not None and max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        cursor_file = self._cursor_file()
        cursor = cursor_file.read()
        total = self.plan.length - cursor.position
        if max_items is not None:
            total = min(total, max_items)
        shards = _ShardsBySubdirectory(self.directory, self.rows_per_shard)
        rejects = RejectLog(self.directory / REJECTS_NAME)
        next_position = cursor.position
        consecutive_rejects = 0

        # One GET per id; a transient failure raises out of the loop. The
        # bar counts ids, so it can end a little short at a plan's end
        # (skipped positions)
        try:
            positions = zip(range(total), self.plan.ids_from(cursor.position))
            for _, (position, item_id) in tqdm(positions, total=total, desc=self.label):
                failures = cursor.failures if position == cursor.position else 0
                match self._fetch_or_record_failure(
                    fetch, cursor_file, rejects, position, item_id, failures
                ):
                    case ItemFound(row, subdirectory):
                        shards.append(subdirectory, json.dumps(row))
                        result.found += 1
                        consecutive_rejects = 0
                    case ItemNotFound():
                        result.not_found += 1
                        consecutive_rejects = 0
                    case ItemRejected(reason):
                        rejects.append(position, item_id, reason)
                        result.rejected += 1
                        consecutive_rejects += 1
                next_position = position + 1
                cursor_file.write(next_position)
                self._check_not_blocked(consecutive_rejects)
            self._finish_if_only_skips_remain(cursor_file, next_position)
        finally:
            _logger.info("%s session: %s", self.label, result)
        return result

    def _fetch_or_record_failure(
        self,
        fetch: Callable[[ItemId], FetchOutcome],
        cursor_file: CursorFile,
        rejects: RejectLog,
        position: int,
        item_id: ItemId,
        failures: int,
    ) -> FetchOutcome:
        """fetch(item_id); on a transient error, record it and re-raise:
        below policy.max_session_failures the cursor keeps position with
        failures + 1 (a warning names self.label and the id); at it, the
        id is rejected ("transient x N: <error>") and the cursor moves
        past it.

        Inputs: fetch, cursor_file, rejects, position, item_id, failures
            (sessions already failed at position).
        Output: fetch's outcome.
        Side effects: on a transient error, rewrites cursor.json and may
            append a reject; logs a warning.
        Exceptions: the transient error, re-raised; OSError."""
        try:
            return fetch(item_id)
        except (requests.RequestException, ValueError) as error:
            failed_sessions = failures + 1
            if failed_sessions < self.policy.max_session_failures:
                _logger.warning(
                    "%s: %s failed (session %d of %d here): %r",
                    self.label,
                    item_id,
                    failed_sessions,
                    self.policy.max_session_failures,
                    error,
                )
                cursor_file.write(position, failed_sessions)
            else:
                rejects.append(
                    position, item_id, f"transient x {failed_sessions}: {error!r}"
                )
                cursor_file.write(position + 1)
            raise

    def _check_not_blocked(self, consecutive_rejects: int) -> None:
        """Inputs: consecutive_rejects (rejects in a row so far).
        Output: none. Side effects: none.
        Exceptions: CrawlBlockedError if it reached
            policy.max_consecutive_rejects."""
        limit = self.policy.max_consecutive_rejects
        if limit is not None and consecutive_rejects >= limit:
            raise CrawlBlockedError(
                f"{self.label}: {consecutive_rejects} rejects in a row; the site "
                "is likely refusing this client (see rejects.jsonl)"
            )

    def _finish_if_only_skips_remain(
        self, cursor_file: CursorFile, next_position: int
    ) -> None:
        """Write the cursor as plan.length if next_position < plan.length
        and plan.ids_from(next_position) yields nothing (only skipped
        positions remain).
        Inputs: cursor_file, next_position. Output: none.
        Side effects: may replace cursor.json. Exceptions: OSError."""
        if next_position >= self.plan.length:
            return
        if next(self.plan.ids_from(next_position), None) is None:
            cursor_file.write(self.plan.length)

    def _cursor_file(self) -> CursorFile:
        """This crawl's CursorFile. Inputs: none. Output: CursorFile.
        Side effects: none. Exceptions: none."""
        return CursorFile(
            self.directory / CURSOR_NAME, self.plan.fingerprint, self.plan.length
        )


class _ShardsBySubdirectory:
    """One JsonlShards per ItemFound.subdirectory (None: the crawl's own
    directory), opened on first use, so a crawl that never lands a
    private row never creates private/."""

    def __init__(self, directory: Path, rows_per_shard: int) -> None:
        """Inputs: directory, rows_per_shard. Output: none.
        Side effects: none. Exceptions: none."""
        self._directory = directory
        self._rows_per_shard = rows_per_shard
        self._shards: dict[str | None, JsonlShards] = {}

    def append(self, subdirectory: str | None, row: str) -> None:
        """Append row to subdirectory's shards.
        Inputs: subdirectory, row. Output: none.
        Side effects: may open (and repair) that directory's shards;
            appends and fsyncs one line. Exceptions: OSError."""
        shards = self._shards.get(subdirectory)
        if shards is None:
            directory = (
                self._directory
                if subdirectory is None
                else self._directory / subdirectory
            )
            shards = JsonlShards(directory, self._rows_per_shard)
            self._shards[subdirectory] = shards
        shards.append(row)
