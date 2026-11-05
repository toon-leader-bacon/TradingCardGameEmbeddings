"""Downloads Archidekt decks (MTG, mostly Commander) one deck id at a
time through Archidekt's open, undocumented read API.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Archidekt's full deck
JSON is ~280 KB for a 100-card deck (prices, card text, combo data), so
phase 2 unwraps each deck by pruning: the kept keys keep the API's own
names and nesting (_KEPT_DECK_KEYS and friends), and nothing is
interpreted. Refining (formats, boards, card matching) is
data_refinement's job. Pruning is close to final, since a re-crawl
takes months, so only bulk (prices, card text, combo data, images) and
the owner are dropped.

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

WALK ORDER (the user's choice): first the ~17,500 decks the sitemap
lists (sitemaps/decks.xml; popular, recently active, listed Commander
decks; a tiny slice, not an index), then every id in [1, newest] in a
fixed spread order (_SpreadOrder), so each prefix is a near-uniform
sample over 2019 to today. Decks created after phase 1 are beyond the
plan; a later plan can cover them.

UNLISTED DECKS (the user's choice): an owner can mark a deck unlisted
(hidden from Archidekt's search, readable by link). They are kept with
their "unlisted" flag, so refinement can leave them out of training.
Nothing about the owner is kept.

    1. phase_1() fetches the sitemap's deck ids and the newest deck id
       (the list ordered by -createdAt) and writes crawl_plan.json. An
       existing plan is kept, so the walk never changes under a running
       crawl; with a cursor but no plan, it refuses (the cursor belongs
       to a lost plan).
    2. phase_2() walks the plan from cursor.json, one paced GET per id,
       for up to max_decks_per_session ids:
         - a deck: one pruned row appended to decks-<nnnnn>.jsonl
           (10,000 rows a shard, gzipped once full);
         - a 404: nothing;
         - a permanent failure (a 4xx other than 404 and 429, or a
           JSON 200 _prune_deck can't read): one row in rejects.jsonl,
           and the walk moves on;
         - a transient failure (network, 5xx, 429 beyond its waits, a
           200 that isn't JSON, as a Cloudflare or maintenance page
           is): raised; the cursor stays on that id, so the next
           session retries it. After _MAX_SESSION_FAILURES sessions fail
           on the same id, that id is rejected ("transient x N") and the
           walk moves on, so one broken deck can't stall the crawl.
       The cursor is replaced after each id, after the row, so a crash
       repeats at most one id (a duplicate row, harmless: deck identity
       downstream is the card multiset).
"""

import gzip
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Iterator

import requests

from tqdm import tqdm

from src.data_retrieval.download_utils import paced_get
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_SITE_URL = "https://archidekt.com"
_USER_AGENT = "TradingCardGameEmbeddings-research/0.1 (non-commercial)"
_REQUEST_TIMEOUT_SECONDS = 60
# A pruned 100-card deck is ~40-60 KB, so a full shard is ~500 MB plain,
# ~100 MB gzipped
_SHARD_ROWS = 10_000
_SHARD_PATTERN = re.compile(r"^decks-(\d{5})\.jsonl(?:\.gz)?$")
_PLAN_NAME = "crawl_plan.json"
_CURSOR_NAME = "cursor.json"
_REJECTS_NAME = "rejects.jsonl"
# Every 4xx is a permanent answer for one id, except 429 (waited out)
_PERMANENT_STATUSES = frozenset(range(400, 500)) - {429}
_SITEMAP_DECK_PATTERN = re.compile(r"<loc>\s*[^<]*?/decks/(\d+)/?\s*</loc>")
_GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
_CURSOR_REPLACE_ATTEMPTS = 5
# Sessions that may fail on one id before it is rejected and skipped
_MAX_SESSION_FAILURES = 3
_CURSOR_REPLACE_BACKOFF_SECONDS = 0.2
_LINE_SCAN_CHUNK_BYTES = 1 << 20

# Pruning: the keys kept at each level, under the API's own names
_KEPT_DECK_KEYS = (
    "id", "name", "createdAt", "updatedAt", "deckFormat", "edhBracket",
    "game", "description", "viewCount", "hasPrimer", "private",
    "unlisted", "theorycrafted", "categories", "deckTags", "customCards",
)  # fmt: skip
_KEPT_CARD_ROW_KEYS = (
    "id", "quantity", "modifier", "categories", "removedCategories",
    "companion", "label", "notes", "customCmc", "createdAt", "deletedAt",
)  # fmt: skip
_KEPT_CARD_KEYS = ("uid", "collectorNumber", "displayName", "releasedAt")
_KEPT_EDITION_KEYS = ("editioncode", "editionname")
_KEPT_ORACLE_CARD_KEYS = ("uid", "name", "layout")


@dataclass(frozen=True)
class _SpreadOrder:
    """A fixed permutation of the ids 1..n: position k -> id
    ((offset + k * stride) mod n) + 1. stride is coprime to n, so every
    id appears exactly once. stride and offset are persisted in the plan
    (not recomputed from n), so a later change to for_range() can't
    reorder a running crawl.

    Invariants (checked in __post_init__): n >= 1, stride >= 1,
    gcd(stride, n) == 1, 0 <= offset < n.
    """

    n: int
    stride: int
    offset: int

    def __post_init__(self) -> None:
        """Exceptions: ValueError if an invariant fails."""
        if self.n < 1:
            raise ValueError(f"spread n must be >= 1, got {self.n}")
        if self.stride < 1 or math.gcd(self.stride, self.n) != 1:
            raise ValueError(
                f"spread stride {self.stride} must be >= 1 and coprime to {self.n}"
            )
        if not 0 <= self.offset < self.n:
            raise ValueError(f"spread offset {self.offset} not in [0, {self.n})")

    @classmethod
    def for_range(cls, n: int) -> "_SpreadOrder":
        """The spread order for 1..n: stride is the first coprime of n at
        or above n / golden ratio (consecutive positions land far apart;
        each prefix is near-uniform over 1..n), offset 0.

        Inputs: n (>= 1). Output: _SpreadOrder.
        Side effects: none. Exceptions: ValueError if n < 1.

        Example:
            >>> _SpreadOrder.for_range(10)
            _SpreadOrder(n=10, stride=7, offset=0)
        """
        if n < 1:
            raise ValueError(f"spread n must be >= 1, got {n}")
        stride = max(1, round(n / _GOLDEN_RATIO))
        while math.gcd(stride, n) != 1:
            stride += 1
        return cls(n, stride, 0)

    def id_at(self, position: int) -> int:
        """The id at position (0 <= position < n).

        Inputs: position. Output: int in [1, n]. Side effects: none.
        Exceptions: IndexError outside [0, n).
        """
        if not 0 <= position < self.n:
            raise IndexError(f"spread position {position} not in [0, {self.n})")
        return (self.offset + position * self.stride) % self.n + 1


@dataclass(frozen=True)
class _CrawlPlan:
    """The walk: sitemap ids first, then the spread order over
    [1, newest_id], skipping sitemap ids. Persisted as crawl_plan.json.

    A position counts every sitemap id and every spread slot (a skipped
    one too), so a position never shifts and one integer is the cursor.
    A sitemap id above newest_id (the gap between the two phase-1 GETs)
    is harmless: it is walked once, in the sitemap part.

    Invariants (checked in __post_init__): sitemap ids positive and
    unique.
    """

    sitemap_ids: tuple[int, ...]
    spread: _SpreadOrder
    _sitemap_id_set: frozenset[int] = field(
        init=False, repr=False, compare=False, default=frozenset()
    )

    def __post_init__(self) -> None:
        """Exceptions: ValueError if an invariant fails."""
        id_set = frozenset(self.sitemap_ids)
        if len(id_set) != len(self.sitemap_ids):
            raise ValueError("plan sitemap ids repeat")
        if any(deck_id < 1 for deck_id in self.sitemap_ids):
            raise ValueError("plan sitemap ids must be positive")
        # Frozen: the cached set is set once, here
        object.__setattr__(self, "_sitemap_id_set", id_set)

    @property
    def length(self) -> int:
        """Total walk positions: len(sitemap_ids) + spread.n."""
        return len(self.sitemap_ids) + self.spread.n

    def ids_from(self, position: int) -> Iterator[tuple[int, int]]:
        """(walk position, deck id) pairs from position to the end,
        skipping spread slots that hold a sitemap id (checked against a
        set of the sitemap ids built once per plan, not per call).

        Inputs: position (>= 0). Output: iterator of (position, id).
        Side effects: none. Exceptions: none.

        Example:
            >>> plan = _CrawlPlan((7,), _SpreadOrder(3, 1, 0))
            >>> list(plan.ids_from(0))
            [(0, 7), (1, 1), (2, 2), (3, 3)]
        """
        sitemap_count = len(self.sitemap_ids)
        for walk_position in range(position, self.length):
            if walk_position < sitemap_count:
                yield walk_position, self.sitemap_ids[walk_position]
                continue
            deck_id = self.spread.id_at(walk_position - sitemap_count)
            if deck_id not in self._sitemap_id_set:
                yield walk_position, deck_id

    @cached_property
    def fingerprint(self) -> str:
        """The SHA-256 of to_json() (hex), stored with the cursor so a
        cursor never walks another plan. Cached: the cursor is written
        once a second. Side effects: none. Exceptions: none."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """The plan as crawl_plan.json text: {"sitemap_ids": [...],
        "spread": {"n", "stride", "offset"}}.
        Output: str. Side effects: none. Exceptions: none."""
        return json.dumps(
            {
                "sitemap_ids": list(self.sitemap_ids),
                "spread": {
                    "n": self.spread.n,
                    "stride": self.spread.stride,
                    "offset": self.spread.offset,
                },
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "_CrawlPlan":
        """Parse crawl_plan.json text.

        Exceptions: ValueError for a malformed plan or a broken
            invariant.
        """
        try:
            raw = json.loads(text)
            sitemap_ids = tuple(_checked_int(item) for item in raw["sitemap_ids"])
            spread = raw["spread"]
            order = _SpreadOrder(
                _checked_int(spread["n"]),
                _checked_int(spread["stride"]),
                _checked_int(spread["offset"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed {_PLAN_NAME}: {error!r}") from error
        return cls(sitemap_ids, order)


@dataclass(frozen=True)
class _DeckFound:
    """A deck, pruned, ready to land."""

    row: dict[str, Any]


@dataclass(frozen=True)
class _DeckNotFound:
    """A 404: deleted, private or never used."""


@dataclass(frozen=True)
class _DeckRejected:
    """A permanent failure for one id: a 4xx other than 404 and 429,
    or a JSON 200 whose body isn't a deck. Recorded in rejects.jsonl;
    the walk moves on. A later pass can re-walk the rejects."""

    reason: str


_DeckFetch = _DeckFound | _DeckNotFound | _DeckRejected


@dataclass
class _SessionCounts:
    """One phase_2() call's outcomes, one count per _DeckFetch variant."""

    found: int = 0
    not_found: int = 0
    rejected: int = 0


def _cut_torn_last_line(path: Path) -> int:
    """Truncate path after its last newline (a crash mid-write leaves a
    partial last line that would merge with the next append) and return
    the number of whole lines it holds. Shared by _DeckShards and the
    rejects file.

    Inputs: path (may not exist: 0, nothing done).
    Output: whole lines in path.
    Side effects: reads path in chunks; may truncate it.
    Exceptions: OSError.
    """
    if not path.is_file():
        return 0
    line_count = 0
    end_of_last_line = 0
    offset = 0
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_LINE_SCAN_CHUNK_BYTES), b""):
            newlines = chunk.count(b"\n")
            if newlines:
                line_count += newlines
                end_of_last_line = offset + chunk.rindex(b"\n") + 1
            offset += len(chunk)
    if end_of_last_line != offset:
        _logger.warning(
            "Cutting a torn last line (%d bytes) off %s",
            offset - end_of_last_line,
            path,
        )
        with open(path, "r+b") as file:
            file.truncate(end_of_last_line)
    return line_count


def _append_line(path: Path, line: str) -> None:
    """Append line plus a newline to path, flushed and fsynced, so a
    cursor written after it never runs ahead of the data on a power loss.

    Inputs: path (Path), line (str, no newline). Output: none.
    Side effects: creates or appends to path; one fsync (about once a
        second in a crawl). Exceptions: OSError.
    """
    # newline="\n": the same bytes on every platform (no CRLF on Windows)
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write(line + "\n")
        file.flush()
        os.fsync(file.fileno())


class _DeckShards:
    """Appends rows to decks-<nnnnn>.jsonl, _SHARD_ROWS rows a shard, and
    gzips each shard once full (decks-<nnnnn>.jsonl.gz; JSON rows
    compress ~5x).

    The files are the only state. Opening:
      - gzips any full plain shard below the newest (a crash between
        filling it and compressing it);
      - takes the newest number; if that shard is already gzipped, the
        next one starts empty; else counts its lines via
        _cut_torn_last_line, which also cuts a last line torn by a crash.
    """

    def __init__(self, directory: Path, rows_per_shard: int = _SHARD_ROWS) -> None:
        """Inputs: directory (created if absent), rows_per_shard (>= 1).
        Side effects: reads the newest shard; may truncate its torn
        last line; may gzip older plain shards. Exceptions: ValueError
        if rows_per_shard < 1; OSError."""
        if rows_per_shard < 1:
            raise ValueError(f"rows_per_shard must be >= 1, got {rows_per_shard}")
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._rows_per_shard = rows_per_shard
        shard_numbers = {
            int(match.group(1))
            for path in directory.iterdir()
            if (match := _SHARD_PATTERN.match(path.name))
        }
        self._shard_number = max(shard_numbers, default=0)

        # Finish any compression a crash interrupted
        for number in sorted(shard_numbers - {self._shard_number}):
            plain_path = self._plain_path(number)
            if plain_path.is_file():
                _gzip_and_remove(plain_path)

        if self._plain_path(self._shard_number).with_suffix(".jsonl.gz").is_file():
            _gzip_and_remove(self._plain_path(self._shard_number))
            self._shard_number += 1
            self._rows_in_shard = 0
        else:
            self._rows_in_shard = _cut_torn_last_line(
                self._plain_path(self._shard_number)
            )
            # Full but not yet gzipped (a crash right after its last row)
            if self._rows_in_shard >= self._rows_per_shard:
                self._rotate()

    def append(self, row: str) -> None:
        """Append row (no newline); once the shard is full, gzip it and
        start the next. Side effects: appends and flushes one line; may
        gzip a shard. Exceptions: OSError."""
        _append_line(self._plain_path(self._shard_number), row)
        self._rows_in_shard += 1
        if self._rows_in_shard >= self._rows_per_shard:
            self._rotate()

    def _rotate(self) -> None:
        """Gzip the current shard and start the next, empty.
        Side effects: writes the .gz, deletes the plain shard.
        Exceptions: OSError."""
        _gzip_and_remove(self._plain_path(self._shard_number))
        self._shard_number += 1
        self._rows_in_shard = 0

    def _plain_path(self, number: int) -> Path:
        """Shard number's plain path.
        Inputs: number (int). Output: Path. Side effects: none."""
        return self._directory / f"decks-{number:05d}.jsonl"


def _gzip_and_remove(plain_path: Path) -> None:
    """Write plain_path.gz, then delete plain_path. The .gz is written
    to a temp file, fsynced, checked to decompress to plain_path's exact
    bytes, and only then moved into place (os.replace) and the plain
    shard deleted: a power loss can't leave a .gz with lost data blocks
    in place of the plain shard. A missing plain_path is a no-op
    (already done).

    Inputs: plain_path (Path). Output: none.
    Side effects: writes and fsyncs the .gz; deletes plain_path.
    Exceptions: OSError; ValueError if the .gz doesn't round-trip (the
        temp file is deleted, the plain shard kept).
    """
    if not plain_path.is_file():
        return
    gz_path = plain_path.with_suffix(".jsonl.gz")
    temp_path = gz_path.with_name(gz_path.name + ".tmp")
    with open(plain_path, "rb") as source, open(temp_path, "wb") as raw_target:
        with gzip.GzipFile(fileobj=raw_target, mode="wb") as target:
            shutil.copyfileobj(source, target, _LINE_SCAN_CHUNK_BYTES)
        raw_target.flush()
        os.fsync(raw_target.fileno())
    if _sha256_of_gzip_content(temp_path) != _sha256_of_file(plain_path):
        temp_path.unlink()
        raise ValueError(f"{temp_path.name} does not round-trip; kept {plain_path}")
    os.replace(temp_path, gz_path)
    plain_path.unlink()


def _sha256_of_file(path: Path) -> str:
    """SHA-256 (hex) of path's bytes. Side effects: reads path.
    Exceptions: OSError."""
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _sha256_of_gzip_content(path: Path) -> str:
    """SHA-256 (hex) of a gzip file's decompressed bytes.
    Side effects: reads path. Exceptions: OSError, gzip.BadGzipFile,
        EOFError for a truncated file."""
    with gzip.open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class ArchidektDeckDownloader(Downloader):
    """Walks Archidekt deck ids and lands one pruned row per deck.

    Single-consumer to src/data_refinement/deck_box/archidekt/.
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
        self._rejects_checked = False

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
        plan_path = self.raw_data_dir / _PLAN_NAME
        if plan_path.is_file():
            return plan_path
        if (self.raw_data_dir / _CURSOR_NAME).is_file():
            raise RuntimeError(
                f"{_CURSOR_NAME} exists but {_PLAN_NAME} does not; restore the "
                "plan or remove the cursor deliberately"
            )

        # Gather the targets
        sitemap_ids = self._sitemap_deck_ids()
        newest_id = self._newest_deck_id()
        plan = _CrawlPlan(tuple(sitemap_ids), _SpreadOrder.for_range(newest_id))
        _logger.info(
            "Archidekt plan: %d sitemap decks, then ids 1..%d (stride %d)",
            len(sitemap_ids),
            newest_id,
            plan.spread.stride,
        )

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.to_json(), encoding="utf-8")
        return plan_path

    def phase_2(self) -> Path:
        """Walk the plan from the cursor, for up to max_decks_per_session
        ids.

        Inputs: none (reads crawl_plan.json and cursor.json).
        Output: raw_data_dir.
        Side effects: one paced GET per walk position (more on a
            retry); appends deck rows to the shards and permanent
            failures to rejects.jsonl; replaces cursor.json after each
            id; a tqdm bar over walk positions; logs the session's
            counts.
        Exceptions: FileNotFoundError if phase_1() never wrote a plan;
            ValueError if the cursor belongs to another plan, or a 200
            body is not JSON; requests.RequestException for a transient
            failure that outlasted its retries. Either transient failure
            leaves the cursor on that id with its failure count raised;
            at _MAX_SESSION_FAILURES the id is rejected and the cursor
            moves past it before the error is raised.

        Example:
            >>> ArchidektDeckDownloader(max_decks_per_session=3600).phase_2()
            PosixPath('data/raw/archidekt')
        """
        plan = _CrawlPlan.from_json(
            (self.raw_data_dir / _PLAN_NAME).read_text(encoding="utf-8")
        )
        cursor = self._read_cursor(plan)
        start = cursor.position
        shards = _DeckShards(self.raw_data_dir)
        counts = _SessionCounts()
        total = plan.length - start
        if self.max_decks_per_session is not None:
            total = min(total, self.max_decks_per_session)

        # Walk, one GET per id; a transient failure raises out of the
        # loop. The bar counts deck ids, so it can end a little short
        # at the plan's end (skipped sitemap slots)
        try:
            walk = zip(range(total), plan.ids_from(start))
            for _, (position, deck_id) in tqdm(
                walk, total=total, desc="Archidekt decks", unit="deck"
            ):
                failures = cursor.failures if position == cursor.position else 0
                try:
                    fetched = self._fetch_deck(deck_id)
                except (requests.RequestException, ValueError) as error:
                    self._record_transient_failure(
                        plan, position, deck_id, failures + 1, error
                    )
                    raise
                match fetched:
                    case _DeckFound(row):
                        shards.append(json.dumps(row))
                        counts.found += 1
                    case _DeckNotFound():
                        counts.not_found += 1
                    case _DeckRejected(reason):
                        self._append_reject(position, deck_id, reason)
                        counts.rejected += 1
                self._write_cursor(plan, position + 1)
                start = position + 1
            # Only skipped slots left: the plan is done
            if next(plan.ids_from(start), None) is None and start < plan.length:
                self._write_cursor(plan, plan.length)
        finally:
            _logger.info("Archidekt session: %s", counts)
        return self.raw_data_dir

    @property
    def _headers(self) -> dict[str, str]:
        """Headers for every request: a plain, honest User-Agent."""
        return {"User-Agent": _USER_AGENT}

    def _record_transient_failure(
        self,
        plan: _CrawlPlan,
        position: int,
        deck_id: int,
        failures: int,
        error: Exception,
    ) -> None:
        """Note a session ending on a transient failure at position: the
        cursor keeps position with the new failure count, or, at
        _MAX_SESSION_FAILURES, the id is rejected and the cursor moves on.

        Inputs: plan, position, deck_id, failures (sessions failed here,
            this one included), error (for the reject's reason).
        Output: none. Side effects: rewrites cursor.json; may append a
            reject. Exceptions: OSError.
        """
        if failures < _MAX_SESSION_FAILURES:
            _logger.warning(
                "Archidekt deck %d failed (session %d of %d here): %r",
                deck_id,
                failures,
                _MAX_SESSION_FAILURES,
                error,
            )
            self._write_cursor(plan, position, failures)
            return
        self._append_reject(position, deck_id, f"transient x {failures}: {error!r}")
        self._write_cursor(plan, position + 1)

    def _get(
        self, url: str, tolerated_statuses: frozenset[int] = frozenset()
    ) -> requests.Response:
        """paced_get with this downloader's limiter, headers and timeout.

        Inputs: url (str), tolerated_statuses (non-2xx returned, not
            raised). Output: requests.Response.
        Side effects: one paced GET (more on a retry or a 429).
        Exceptions: as paced_get."""
        return paced_get(
            url,
            rate_limiter=self.rate_limiter,
            headers=self._headers,
            timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
            tolerated_statuses=tolerated_statuses,
        )

    def _sitemap_deck_ids(self) -> list[int]:
        """The deck ids sitemaps/decks.xml lists, in file order, without
        repeats.
        Inputs: none. Output: list[int]. Side effects: one paced GET.
        Exceptions: requests.RequestException; ValueError if none."""
        text = self._get(f"{self.site_url}/sitemaps/decks.xml").text
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
        listing = self._get(f"{self.site_url}/api/decks/v3/?orderBy=-createdAt").json()
        deck_ids = [int(deck["id"]) for deck in listing.get("results", [])]
        if not deck_ids:
            raise ValueError("Archidekt's newest-decks list has no results")
        return max(deck_ids)

    def _fetch_deck(self, deck_id: int) -> _DeckFetch:
        """GET /api/decks/<id>/ through download_utils.paced_get with
        tolerated_statuses=_PERMANENT_STATUSES, so every permanent answer
        comes back as a response and anything raised is transient; then
        classify the answer.

        Inputs: deck_id. Output: _DeckFound (pruned via _prune_deck),
            _DeckNotFound (404), or _DeckRejected (another tolerated 4xx,
            or a JSON body _prune_deck can't read).
        Side effects: one paced GET (more on a retry).
        Exceptions: requests.RequestException from paced_get (network,
            5xx, 429 past its waits); ValueError for a 200 whose body is
            not JSON (a challenge or maintenance page: transient, the
            cursor stays).
        """
        response = self._get(
            f"{self.site_url}/api/decks/{deck_id}/", _PERMANENT_STATUSES
        )
        if response.status_code == 404:
            return _DeckNotFound()
        if response.status_code in _PERMANENT_STATUSES:
            return _DeckRejected(f"HTTP {response.status_code}")
        try:
            payload = json.loads(response.text)
        except ValueError as error:
            raise ValueError(
                f"deck {deck_id}: HTTP {response.status_code} body is not JSON: "
                f"{response.text[:200]!r}"
            ) from error
        try:
            return _DeckFound(_prune_deck(payload))
        except (KeyError, TypeError) as error:
            return _DeckRejected(f"unreadable deck: {error!r}")

    def _append_reject(self, position: int, deck_id: int, reason: str) -> None:
        """Append {"position", "id", "reason"} to rejects.jsonl, after
        _cut_torn_last_line on the session's first reject.
        Side effects: appends and flushes one line. Exceptions: OSError."""
        rejects_path = self.raw_data_dir / _REJECTS_NAME
        if not self._rejects_checked:
            _cut_torn_last_line(rejects_path)
            self._rejects_checked = True
        _append_line(
            rejects_path,
            json.dumps({"position": position, "id": deck_id, "reason": reason}),
        )

    def _read_cursor(self, plan: _CrawlPlan) -> "_Cursor":
        """cursor.json, or position 0 with no failures if absent. A cursor
        written before failure counts existed reads as 0 failures.

        Inputs: plan (the cursor must belong to it). Output: _Cursor.
        Side effects: reads cursor.json.
        Exceptions: ValueError for a malformed cursor, one whose plan
            fingerprint isn't plan's, or a position outside
            [0, plan.length]."""
        cursor_path = self.raw_data_dir / _CURSOR_NAME
        if not cursor_path.is_file():
            return _Cursor(0, 0)
        try:
            cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
            position = _checked_int(cursor["position"])
            failures = _checked_int(cursor.get("failures", 0))
            plan_fingerprint = cursor["plan"]
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"malformed {_CURSOR_NAME}: {error!r}") from error
        if plan_fingerprint != plan.fingerprint:
            raise ValueError(f"{_CURSOR_NAME} belongs to another {_PLAN_NAME}")
        if not 0 <= position <= plan.length:
            raise ValueError(
                f"{_CURSOR_NAME} position {position} not in [0, {plan.length}]"
            )
        return _Cursor(position, failures)

    def _write_cursor(self, plan: _CrawlPlan, position: int, failures: int = 0) -> None:
        """Replace cursor.json ({"position", "failures", "plan"})
        atomically: a temp file, then os.replace, retried briefly on
        PermissionError (Windows file locks from antivirus or the
        indexer).

        Inputs: plan, position (next walk position), failures (sessions
            failed at position). Output: none.
        Side effects: replaces cursor.json.
        Exceptions: OSError once the retries are spent."""
        cursor_path = self.raw_data_dir / _CURSOR_NAME
        temp_path = cursor_path.with_name(_CURSOR_NAME + ".tmp")
        temp_path.write_text(
            json.dumps(
                {"position": position, "failures": failures, "plan": plan.fingerprint}
            ),
            encoding="utf-8",
        )
        for attempt in range(_CURSOR_REPLACE_ATTEMPTS):
            try:
                os.replace(temp_path, cursor_path)
                return
            except PermissionError:
                if attempt == _CURSOR_REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_CURSOR_REPLACE_BACKOFF_SECONDS)


@dataclass(frozen=True)
class _Cursor:
    """cursor.json: the next walk position, and how many sessions have
    failed on it."""

    position: int
    failures: int


def _checked_int(value: Any) -> int:
    """value if it is an int (not a bool).
    Inputs: value (a parsed JSON value). Output: int. Side effects: none.
    Exceptions: TypeError otherwise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an int, got {value!r}")
    return value


def _picked(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The entries of source under keys, in keys' order, skipping absent
    ones. Side effects: none. Exceptions: none."""
    return {key: source[key] for key in keys if key in source}


def _prune_deck(payload: dict[str, Any]) -> dict[str, Any]:
    """One deck's JSON pruned to the kept keys, names and nesting
    unchanged: the deck's _KEPT_DECK_KEYS, and "cards" with each row's
    _KEPT_CARD_ROW_KEYS plus "card" holding _KEPT_CARD_KEYS,
    "edition" (_KEPT_EDITION_KEYS) and "oracleCard"
    (_KEPT_ORACLE_CARD_KEYS). A kept key the payload lacks is left out,
    not invented. Values are copied, never interpreted.

    Inputs: payload (one deck's JSON). Output: the pruned dict.
    Side effects: none.
    Exceptions: KeyError if payload lacks "id" or "cards", or a card row
        lacks "card"; TypeError if they have the wrong JSON type.

    Example:
        >>> _prune_deck({"id": 1, "cards": [], "owner": {"id": 9}})
        {'id': 1, 'cards': []}
    """
    if not isinstance(payload, dict):
        raise TypeError(f"deck payload is {type(payload).__name__}, not an object")
    payload["id"]  # required: KeyError if absent
    raw_cards = payload["cards"]
    if not isinstance(raw_cards, list):
        raise TypeError(f"deck cards is {type(raw_cards).__name__}, not a list")

    result = _picked(payload, _KEPT_DECK_KEYS)
    result["cards"] = [_pruned_card_row(entry) for entry in raw_cards]
    return result


def _pruned_card_row(entry: Any) -> dict[str, Any]:
    """One "cards" row pruned (see _prune_deck).
    Exceptions: KeyError if it lacks "card"; TypeError for a non-object
        row or card."""
    if not isinstance(entry, dict):
        raise TypeError(f"card row is {type(entry).__name__}, not an object")
    card = entry["card"]
    if not isinstance(card, dict):
        raise TypeError(f"card is {type(card).__name__}, not an object")

    result = _picked(entry, _KEPT_CARD_ROW_KEYS)
    kept_card = _picked(card, _KEPT_CARD_KEYS)
    for key, kept_keys in (
        ("edition", _KEPT_EDITION_KEYS),
        ("oracleCard", _KEPT_ORACLE_CARD_KEYS),
    ):
        nested = card.get(key)
        if isinstance(nested, dict):
            kept_card[key] = _picked(nested, kept_keys)
    result["card"] = kept_card
    return result
