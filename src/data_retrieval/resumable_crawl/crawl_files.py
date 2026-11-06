"""The on-disk state of a resumable crawl: deck shards, the rejects log
and the cursor, plus the durable line-append helpers they share.

  - decks-<nnnnn>.jsonl, rows_per_shard rows a shard, gzipped once full
    (decks-<nnnnn>.jsonl.gz): JsonlShards;
  - rejects.jsonl, one {"position", "id", "reason"} row per id given up
    on: RejectLog;
  - cursor.json, {"position", "failures", "plan"}: CursorFile.

Every append is fsynced, so a cursor written after it never runs ahead
of the data on a power loss.
"""

import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.data_retrieval.download_utils import require_int

_logger = logging.getLogger(__name__)

SHARD_ROWS = 10_000
CURSOR_NAME = "cursor.json"
REJECTS_NAME = "rejects.jsonl"
_SHARD_PATTERN = re.compile(r"^decks-(\d{5})\.jsonl(?:\.gz)?$")
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_SECONDS = 0.2
_LINE_SCAN_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Cursor:
    """cursor.json: the next walk position, and how many sessions have
    failed on it."""

    position: int
    failures: int


class CursorFile:
    """cursor.json for one plan."""

    def __init__(self, path: Path, plan_fingerprint: str, plan_length: int) -> None:
        """Inputs: path (cursor.json), plan_fingerprint, plan_length.
        Output: none. Side effects: none. Exceptions: none."""
        self._path = path
        self._plan_fingerprint = plan_fingerprint
        self._plan_length = plan_length

    def read(self) -> Cursor:
        """The cursor, or Cursor(0, 0) if absent; a missing "failures"
        reads as 0.

        Inputs: none. Output: Cursor. Side effects: reads path.
        Exceptions: ValueError for a malformed cursor, another plan's
            fingerprint, or a position outside [0, plan_length].

        Example:
            >>> CursorFile(Path("cursor.json"), "ab12", 10).read()
            Cursor(position=0, failures=0)
        """
        if not self._path.is_file():
            return Cursor(0, 0)
        try:
            cursor = json.loads(self._path.read_text(encoding="utf-8"))
            position = require_int(cursor["position"])
            failures = require_int(cursor.get("failures", 0))
            plan_fingerprint = cursor["plan"]
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"malformed {self._path.name}: {error!r}") from error
        if plan_fingerprint != self._plan_fingerprint:
            raise ValueError(f"{self._path.name} belongs to another plan")
        if not 0 <= position <= self._plan_length:
            raise ValueError(
                f"{self._path.name} position {position} not in "
                f"[0, {self._plan_length}]"
            )
        return Cursor(position, failures)

    def write(self, position: int, failures: int = 0) -> None:
        """Replace cursor.json ({"position", "failures", "plan"})
        atomically: a temp file, then os.replace, retried briefly on
        PermissionError (Windows file locks from antivirus or the
        indexer).

        Inputs: position (next walk position), failures (sessions failed
            at position). Output: none. Side effects: replaces path.
        Exceptions: OSError once the retries are spent.

        Example:
            >>> CursorFile(Path("cursor.json"), "ab12", 10).write(3)
        """
        write_text_atomically(
            self._path,
            json.dumps(
                {
                    "position": position,
                    "failures": failures,
                    "plan": self._plan_fingerprint,
                }
            ),
        )


def write_text_atomically(path: Path, text: str) -> None:
    """Replace path with text (UTF-8, LF newlines) via a temp file,
    flushed and fsynced, then os.replace, so neither a reader nor a power
    loss leaves a part-written file; the replace is retried briefly on
    PermissionError (Windows file locks from antivirus or the indexer).

    Inputs: path, text. Output: none.
    Side effects: writes and fsyncs path.tmp, then replaces path.
    Exceptions: OSError once the retries are spent.

    Example:
        >>> write_text_atomically(Path("cursor.json"), '{"position": 3}')
    """
    temp_path = path.with_name(path.name + ".tmp")
    # newline="\n": the same bytes on every platform (no CRLF on Windows)
    with open(temp_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_SECONDS)


def refuse_orphan_cursor(directory: Path, plan_name: str) -> None:
    """For a phase 1 about to write a new plan: refuse if a cursor exists
    without its plan (it belongs to a lost plan).

    Inputs: directory, plan_name (the plan's file name, for the message).
    Output: none. Side effects: checks for cursor.json.
    Exceptions: RuntimeError if directory/cursor.json exists.

    Example:
        >>> refuse_orphan_cursor(Path("data/raw/moxfield"), "crawl_plan.tsv")
    """
    if (directory / CURSOR_NAME).is_file():
        raise RuntimeError(
            f"{CURSOR_NAME} exists but {plan_name} does not; restore the "
            "plan or remove the cursor deliberately"
        )


class RejectLog:
    """rejects.jsonl."""

    def __init__(self, path: Path) -> None:
        """Inputs: path. Output: none. Side effects: none (the torn-line
        check runs on the first append). Exceptions: none."""
        self._path = path
        self._checked = False

    def append(self, position: int, item_id: int | str, reason: str) -> None:
        """Append {"position", "id", "reason"}, after cut_torn_last_line on
        this log's first append.

        Inputs: position, item_id, reason. Output: none.
        Side effects: may truncate a torn last line; appends and fsyncs
            one line. Exceptions: OSError.

        Example:
            >>> RejectLog(Path("rejects.jsonl")).append(7, 4242, "HTTP 410")
        """
        if not self._checked:
            cut_torn_last_line(self._path)
            self._checked = True
        append_line(
            self._path,
            json.dumps({"position": position, "id": item_id, "reason": reason}),
        )


class JsonlShards:
    """Appends rows to decks-<nnnnn>.jsonl, rows_per_shard rows a shard,
    and gzips each shard once full (decks-<nnnnn>.jsonl.gz; JSON rows
    compress ~5x).

    The files are the only state. Opening:
      - gzips any full plain shard below the newest (a crash between
        filling it and compressing it);
      - takes the newest number; if that shard is already gzipped, the
        next one starts empty; else counts its lines via
        cut_torn_last_line, which also cuts a last line torn by a crash,
        and rotates it if it is already full.
    """

    def __init__(self, directory: Path, rows_per_shard: int = SHARD_ROWS) -> None:
        """Inputs: directory (created if absent), rows_per_shard (>= 1).
        Output: none. Side effects: reads the newest shard; may truncate
        its torn last line; may gzip and delete plain shards.
        Exceptions: ValueError if rows_per_shard < 1; OSError."""
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
            gzip_and_remove(self._plain_path(number))

        if self._plain_path(self._shard_number).with_suffix(".jsonl.gz").is_file():
            gzip_and_remove(self._plain_path(self._shard_number))
            self._shard_number += 1
            self._rows_in_shard = 0
        else:
            self._rows_in_shard = cut_torn_last_line(
                self._plain_path(self._shard_number)
            )
            # Full but not yet gzipped (a crash right after its last row)
            if self._rows_in_shard >= self._rows_per_shard:
                self._rotate()

    def append(self, row: str) -> None:
        """Append row (no newline); once the shard is full, gzip it and
        start the next.

        Inputs: row (one JSON line). Output: none.
        Side effects: appends and fsyncs one line; may gzip a shard and
            delete its plain file. Exceptions: OSError.

        Example:
            >>> JsonlShards(Path("data/raw/moxfield")).append('{"publicId": "x"}')
        """
        append_line(self._plain_path(self._shard_number), row)
        self._rows_in_shard += 1
        if self._rows_in_shard >= self._rows_per_shard:
            self._rotate()

    def _rotate(self) -> None:
        """Gzip the current shard and start the next, empty.
        Inputs: none. Output: none.
        Side effects: writes the .gz, deletes the plain shard.
        Exceptions: OSError."""
        gzip_and_remove(self._plain_path(self._shard_number))
        self._shard_number += 1
        self._rows_in_shard = 0

    def _plain_path(self, number: int) -> Path:
        """Shard number's plain path.
        Inputs: number. Output: Path. Side effects: none."""
        return self._directory / f"decks-{number:05d}.jsonl"


def cut_torn_last_line(path: Path) -> int:
    """Truncate path after its last newline (a crash mid-write leaves a
    partial last line that would merge with the next append) and return
    the number of whole lines it holds.

    Inputs: path (may not exist: 0, nothing done).
    Output: whole lines in path.
    Side effects: reads path in chunks; may truncate it; logs a warning
        when it does.
    Exceptions: OSError.

    Example:
        >>> cut_torn_last_line(Path("rejects.jsonl"))
        12
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


def append_line(path: Path, line: str) -> None:
    """Append line plus a newline (LF on every platform) to path, flushed
    and fsynced, so a cursor written after it never runs ahead of the
    data on a power loss.

    Inputs: path, line (no newline). Output: none.
    Side effects: creates or appends to path; one fsync.
    Exceptions: OSError.

    Example:
        >>> append_line(Path("rejects.jsonl"), '{"position": 1}')
    """
    # newline="\n": the same bytes on every platform (no CRLF on Windows)
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write(line + "\n")
        file.flush()
        os.fsync(file.fileno())


def gzip_and_remove(plain_path: Path) -> None:
    """Write plain_path.gz, then delete plain_path. The .gz is written to
    a temp file, fsynced, checked to decompress to plain_path's exact
    bytes, and only then moved into place (os.replace) and the plain
    file deleted: a power loss can't leave a .gz with lost data blocks in
    place of the plain file. A missing plain_path is a no-op (already
    done).

    Inputs: plain_path. Output: none.
    Side effects: writes and fsyncs the .gz; deletes plain_path.
    Exceptions: OSError; ValueError if the .gz doesn't round-trip (the
        temp file is deleted, the plain file kept).

    Example:
        >>> gzip_and_remove(Path("decks-00000.jsonl"))
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
    """SHA-256 (hex) of path's bytes. Inputs: path. Output: str.
    Side effects: reads path. Exceptions: OSError."""
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _sha256_of_gzip_content(path: Path) -> str:
    """SHA-256 (hex) of a gzip file's decompressed bytes. Inputs: path.
    Output: str. Side effects: reads path.
    Exceptions: OSError, gzip.BadGzipFile, EOFError for a truncated file."""
    with gzip.open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
