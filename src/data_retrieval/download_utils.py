"""Shared GET-with-retries helpers: stream a response to a file, or
return its body as text.

Shared across data_retrieval sources — download_to_file() was
extracted after the same stream-to-disk-with-cleanup block showed up
independently in three downloaders (Scryfall, Pokemon TCG,
HearthstoneJSON), each with its own copy of the chunk-size/timeout
constants. Placed at this shared level for the same reason as
rate_limiter.py: no source-specific coupling, and duplicated magic
numbers across three copies risk silently drifting apart. See
src/data_retrieval/README.md.

Retries (added once a second source needed the same retry-on-failure
loop as PlayGwentDownloader's private _get_with_retries) are built in
here rather than left to each caller. download_to_string() was added
alongside it once a two-phase downloader (sts_gg) needed to inspect a
paginated list endpoint's parsed body before deciding whether to keep
paging — a case download_to_file's "URL in, file on disk out" contract
can't serve directly. Both functions share the same attempt-counting/
backoff loop via _call_with_retries() rather than duplicating it, since
the only real difference between them (streaming straight to disk vs.
buffering the whole body as text) lives entirely inside each one's own
attempt closure. See src/data_retrieval/TODO.md's note on migrating
other sources' private retry loops onto these shared functions.
"""

import time
from pathlib import Path
from typing import Callable, TypeVar

import requests

_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_CHUNK_SIZE_BYTES = (
    1024 * 1024
)  # 1 MiB, so large responses never sit fully in memory
_DEFAULT_MAX_ATTEMPTS = 3  # 1 initial try + up to 2 retries
_RETRY_BACKOFF_SECONDS = 2.0

_AttemptResult = TypeVar("_AttemptResult")


def _call_with_retries(
    attempt: Callable[[], _AttemptResult], *, max_attempts: int
) -> _AttemptResult:
    """Call attempt(), retrying up to max_attempts total times if it
    raises requests.RequestException, with a fixed
    _RETRY_BACKOFF_SECONDS pause between attempts.

    Private helper — shared by download_to_file() and
    download_to_string(), this module's two GET helpers, so both retry
    identically without duplicating the attempt-counting/backoff loop.
    Each caller's attempt() closure owns whatever's specific to it
    (streamed writing plus partial-file cleanup for download_to_file,
    plain body buffering for download_to_string) — this function only
    knows about retrying a zero-arg callable.

    Inputs:
        attempt: zero-arg callable performing one full attempt end to
            end, including any side effects that must be redone from
            scratch on retry (e.g. re-opening the destination file).
        max_attempts: total attempts before giving up.
    Output: attempt()'s return value, from whichever attempt succeeds.
    Side effects: whatever attempt() does, once per attempt; sleeps
        _RETRY_BACKOFF_SECONDS between attempts (not after the last
        one).
    Exceptions: raises the last attempt's requests.RequestException
        once max_attempts attempts have all failed. Any other exception
        attempt() raises propagates immediately, without retrying.
    """
    last_error: requests.RequestException | None = None

    for attempt_index in range(max_attempts):
        try:
            return attempt()
        except requests.RequestException as error:
            last_error = error
            if attempt_index < max_attempts - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS)

    assert last_error is not None  # loop always runs >= 1 iteration
    raise last_error


def download_to_file(
    url: str,
    destination_path: Path,
    *,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    chunk_size_bytes: int = _DEFAULT_CHUNK_SIZE_BYTES,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> None:
    """Stream a GET response to destination_path, in chunks, retrying
    on failure.

    Inputs:
        url: URL to GET.
        destination_path: file path the response body is written to.
        timeout_seconds: request timeout, in seconds.
        chunk_size_bytes: size of each streamed write, in bytes.
        max_attempts: total GET attempts before giving up (1 initial
            try plus up to max_attempts - 1 retries), with a fixed
            _RETRY_BACKOFF_SECONDS pause between attempts. Defaults to
            _DEFAULT_MAX_ATTEMPTS (3).
    Output: none.
    Side effects: one network request per attempt (up to max_attempts);
        sleeps _RETRY_BACKOFF_SECONDS between attempts (not after the
        last one); creates destination_path's parent directory if
        missing; writes destination_path.
    Exceptions: raises the last attempt's requests.RequestException
        (e.g. connection error, timeout, non-2xx response) once
        max_attempts attempts have all failed, or raises on failure to
        write the file. Any partially-written file is removed before
        the exception propagates.

    Example:
        >>> download_to_file(
        ...     "https://example.com/file.zip",
        ...     Path("data/raw/example/file.zip"),
        ... )
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    def _attempt() -> None:
        try:
            with requests.get(url, stream=True, timeout=timeout_seconds) as response:
                response.raise_for_status()
                with open(destination_path, "wb") as destination_file:
                    for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                        destination_file.write(chunk)
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise

    _call_with_retries(_attempt, max_attempts=max_attempts)


def download_to_string(
    url: str,
    *,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> str:
    """GET url and return its response body as text, retrying on
    failure.

    For a caller that needs to inspect a response's body before
    deciding what to do next (e.g. read a pagination cursor/total out
    of a listing endpoint, before deciding whether to keep paging) —
    download_to_file's "URL in, file on disk out" contract can't serve
    that, since it never hands the body back. Returns raw text, not a
    parsed object: some responses are JSON, some (e.g. play_gwent's
    guide detail pages) are HTML with a payload embedded inside, and
    this container's own scope (src/data_retrieval/README.md) is
    fetching, not parsing — callers that want JSON call json.loads() on
    the result themselves, same as every existing per-page/per-item
    JSON consumer in this container already does with
    response.json()/json.loads(response.text).

    Inputs:
        url: URL to GET.
        timeout_seconds: request timeout, in seconds.
        max_attempts: total GET attempts before giving up (1 initial
            try plus up to max_attempts - 1 retries), with a fixed
            _RETRY_BACKOFF_SECONDS pause between attempts. Defaults to
            _DEFAULT_MAX_ATTEMPTS (3).
    Output: the response body decoded as text (requests.Response.text).
    Side effects: one network request per attempt (up to max_attempts);
        sleeps _RETRY_BACKOFF_SECONDS between attempts (not after the
        last one).
    Exceptions: raises the last attempt's requests.RequestException
        (e.g. connection error, timeout, non-2xx response) once
        max_attempts attempts have all failed.

    Example:
        >>> body = download_to_string("https://example.com/page.json")
    """

    def _attempt() -> str:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        return response.text

    return _call_with_retries(_attempt, max_attempts=max_attempts)
