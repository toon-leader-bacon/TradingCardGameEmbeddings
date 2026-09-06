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

read_manifest()/append_with_manifest() were extracted once the same
"read a manifest file's ids, else empty set" plus "append a data row,
then its manifest row, flushing both" shape showed up independently in
three two-phase downloaders' phase_2() methods (STSGGRunDownloader,
PlayGwentDownloader, PitchstackDeckDownloader) — the rule-of-three
case, same as download_to_file/download_to_string themselves.

download_to_file()/download_to_string() both gained an optional
`headers` param once FabtcgDecklistDownloader needed one: fabtcg.com's
WAF 403s the default python-requests User-Agent on every endpoint it
uses, but 200s an ordinary browser-style one. Backwards compatible —
every existing caller keeps passing nothing, which still means
"requests' own default headers," identical to before this param
existed.
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
    headers: dict[str, str] | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    chunk_size_bytes: int = _DEFAULT_CHUNK_SIZE_BYTES,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> None:
    """Stream a GET response to destination_path, in chunks, retrying
    on failure.

    Inputs:
        url: URL to GET.
        destination_path: file path the response body is written to.
        headers: extra HTTP headers to send with every attempt (e.g. a
            User-Agent a source's server requires). None (the default)
            sends requests' own default headers, identical to this
            function's behavior before this param existed.
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
            with requests.get(
                url, stream=True, timeout=timeout_seconds, headers=headers
            ) as response:
                response.raise_for_status()
                with open(destination_path, "wb") as destination_file:
                    for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                        destination_file.write(chunk)
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise

    _call_with_retries(_attempt, max_attempts=max_attempts)


def read_manifest(manifest_path: Path) -> set[str]:
    """Read a manifest file's recorded ids, or an empty set if it
    doesn't exist yet.

    Shared across data_retrieval sources — extracted after the same
    read-if-exists-else-empty-set block showed up independently in
    three downloaders (STSGGRunDownloader, PlayGwentDownloader,
    PitchstackDeckDownloader), each pairing it with a JSONL data file
    resumed via append_with_manifest() below. Placed at this shared
    level for the same rule-of-three reason as download_to_file/
    download_to_string — see this module's docstring.

    Inputs:
        manifest_path: path to a manifest file, one id per line.
    Output: recorded ids, as a set (order doesn't matter — this is
        purely a fast "already downloaded" membership check).
    Side effects: reads manifest_path if it exists.
    Exceptions: none expected beyond filesystem errors.

    Example:
        >>> already_downloaded = read_manifest(Path("data/raw/example/manifest.txt"))
    """
    if not manifest_path.exists():
        return set()

    return {
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_with_manifest(
    data_path: Path, manifest_path: Path, data_line: str, manifest_id: str
) -> None:
    """Append data_line to data_path, then manifest_id to
    manifest_path, flushing each write immediately.

    Shared across data_retrieval sources — see read_manifest() above
    for the extraction rationale. data_line is appended BEFORE
    manifest_id, deliberately: a crash between the two leaves a data
    row with no matching manifest entry (a harmless, self-correcting
    gap — the next run just re-fetches and re-appends that same item,
    producing a duplicate line a downstream reader can de-dup on)
    rather than the reverse failure mode, where the manifest would
    claim an item is done while its data row was never written — a
    silent, undetectable loss. This ordering is the entire reason this
    function exists rather than each caller writing two open()/write()
    calls inline: centralizing it here means every caller gets it
    right by construction, not by each one remembering the rule.

    Inputs:
        data_path: JSONL (or similar line-oriented) file data_line is
            appended to, exactly as given — this function has no
            opinion on its contents' format.
        manifest_path: manifest file manifest_id is appended to, after
            data_path's write completes.
        data_line: one line's worth of content, without a trailing
            newline (this function adds it).
        manifest_id: the id being recorded as done, without a trailing
            newline (this function adds it).
    Output: none.
    Side effects: appends one line to data_path, flushing immediately;
        then appends one line to manifest_path, flushing immediately.
    Exceptions: raises on failure to write either file.

    Example:
        >>> append_with_manifest(
        ...     Path("data/raw/example/items.jsonl"),
        ...     Path("data/raw/example/manifest.txt"),
        ...     json.dumps(payload),
        ...     item_id,
        ... )
    """
    with open(data_path, "a", encoding="utf-8") as data_file:
        data_file.write(data_line + "\n")
        data_file.flush()

    with open(manifest_path, "a", encoding="utf-8") as manifest_file:
        manifest_file.write(manifest_id + "\n")
        manifest_file.flush()


def download_to_string(
    url: str,
    *,
    headers: dict[str, str] | None = None,
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
        headers: extra HTTP headers to send with every attempt (e.g. a
            User-Agent a source's server requires). None (the default)
            sends requests' own default headers, identical to this
            function's behavior before this param existed.
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
        response = requests.get(url, timeout=timeout_seconds, headers=headers)
        response.raise_for_status()
        return response.text

    return _call_with_retries(_attempt, max_attempts=max_attempts)
