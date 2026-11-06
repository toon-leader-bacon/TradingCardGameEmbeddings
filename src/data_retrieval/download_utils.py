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
backoff loop via call_with_retries() rather than duplicating it, since
the only real difference between them (streaming straight to disk vs.
buffering the whole body as text) lives entirely inside each one's own
attempt closure. See src/data_retrieval/TODO.md's note on migrating
other sources' private retry loops onto these shared functions.

call_with_retries() itself is exported (not just an internal helper of
this module) for a caller whose attempt doesn't fit either
download_to_file's or download_to_string's contract but still wants
the same attempt-counting/backoff behavior — e.g.
SpireCodexRunDownloader._fetch_page, which needs a response header
(X-Next-Cursor) alongside the streamed-to-disk body, something neither
function surfaces.

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

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import requests

from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_CHUNK_SIZE_BYTES = (
    1024 * 1024
)  # 1 MiB, so large responses never sit fully in memory
_DEFAULT_MAX_ATTEMPTS = 3  # 1 initial try + up to 2 retries
_RETRY_BACKOFF_SECONDS = 2.0
_DEFAULT_MAX_RATE_LIMITED_WAITS = 5
_DEFAULT_RATE_LIMITED_WAIT_SECONDS = 60.0  # a 429 without Retry-After
_MAX_RATE_LIMITED_WAIT_SECONDS = 900.0  # caps a huge Retry-After

_AttemptResult = TypeVar("_AttemptResult")


def call_with_retries(
    attempt: Callable[[], _AttemptResult], *, max_attempts: int
) -> _AttemptResult:
    """Call attempt(), retrying up to max_attempts total times if it
    raises requests.RequestException, with a fixed
    _RETRY_BACKOFF_SECONDS pause between attempts.

    Shared by download_to_file() and download_to_string(), this
    module's two GET helpers, so both retry identically without
    duplicating the attempt-counting/backoff loop. Each caller's
    attempt() closure owns whatever's specific to it (streamed writing
    plus partial-file cleanup for download_to_file, plain body
    buffering for download_to_string) — this function only knows about
    retrying a zero-arg callable. Also exported for a caller whose
    attempt doesn't fit either of those two contracts (see this
    module's docstring) but still wants the same retry behavior.

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

    Example:
        >>> def _attempt() -> str:
        ...     response = requests.get(url, timeout=60)
        ...     response.raise_for_status()
        ...     return response.text
        >>> body = call_with_retries(_attempt, max_attempts=3)
    """
    last_error: requests.RequestException | None = None

    for attempt_index in range(max_attempts):
        try:
            return attempt()
        except requests.RequestException as error:
            last_error = error
            # Logged so a crawl's throttling (5xx, resets) shows in its log
            _logger.warning(
                "Attempt %d/%d failed: %s", attempt_index + 1, max_attempts, error
            )
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

    call_with_retries(_attempt, max_attempts=max_attempts)


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

    return call_with_retries(_attempt, max_attempts=max_attempts)


def paced_get(
    url: str,
    *,
    rate_limiter: RateLimiter,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    tolerated_statuses: frozenset[int] = frozenset(),
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    max_rate_limited_waits: int = _DEFAULT_MAX_RATE_LIMITED_WAITS,
    follow_redirects: bool = True,
) -> requests.Response:
    """GET url, paced, returning the response when its status is 2xx or
    in tolerated_statuses; retrying network errors and every other
    non-2xx status (call_with_retries; a 5xx, but also an untolerated
    4xx); waiting out a 429 by its Retry-After header.

    For a crawl where a "not found" is an ordinary answer (Archidekt:
    about half of all deck ids are 404s), so it must not be raised into
    call_with_retries and retried. A 429 is not retried through
    call_with_retries either: its fixed backoff falls inside the
    server's rate-limit window (the reason limitless_play waits 429s
    out itself). hearthstonetopdecks, fabtcgmeta_online_decks and
    limitless_play carry older copies of these rules (TODO.md).

    Inputs:
        url: URL to GET.
        rate_limiter: waited on before every request, retries included.
        headers, timeout_seconds: as download_to_string.
        tolerated_statuses: non-2xx statuses returned, not raised
            (empty by default). Must not hold 429, which is waited out.
            A caller that tolerates every permanent answer (Archidekt:
            all 4xx but 429) knows anything raised is transient.
        max_attempts: total attempts against network errors and 5xx.
        max_rate_limited_waits: 429s waited out for this one GET before
            raising (>= 0). A Retry-After is capped at
            _MAX_RATE_LIMITED_WAIT_SECONDS.
        follow_redirects: False returns a 3xx as it is (raise_for_status
            passes 3xx), so a caller can pace the redirect's GET itself;
            True lets requests follow it, unpaced.
    Output: the requests.Response (2xx, a tolerated status, or a 3xx
        when not following redirects).
    Side effects: one or more paced requests; sleeps on a 429 (the
        Retry-After seconds, or a minute when absent) and between
        retries.
    Exceptions: ValueError if tolerated_statuses holds 429 or
        max_rate_limited_waits < 0;
        requests.HTTPError for any other status, or a 429 past
        max_rate_limited_waits; requests.RequestException once
        max_attempts attempts have failed.

    Example:
        >>> response = paced_get(url, rate_limiter=RateLimiter(60),
        ...                      tolerated_statuses=frozenset({404}))
        >>> response.status_code
        404
    """
    if 429 in tolerated_statuses:
        raise ValueError("tolerated_statuses must not hold 429; it is waited out")
    if max_rate_limited_waits < 0:
        raise ValueError(
            f"max_rate_limited_waits must be >= 0, got {max_rate_limited_waits}"
        )

    def _attempt() -> requests.Response:
        rate_limiter.wait()
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers=headers,
            allow_redirects=follow_redirects,
        )
        # A 429 is returned, not raised: call_with_retries would retry it
        # after a fixed backoff, inside the server's window
        if (
            response.status_code != 429
            and response.status_code not in tolerated_statuses
        ):
            response.raise_for_status()
        return response

    # Request, waiting out each 429 except the last
    for waits_so_far in range(max_rate_limited_waits + 1):
        response = call_with_retries(_attempt, max_attempts=max_attempts)
        if response.status_code != 429:
            return response
        if waits_so_far == max_rate_limited_waits:
            break
        wait_seconds = _retry_after_seconds(response)
        _logger.warning("Rate limited by %s; waiting %.0f s", url, wait_seconds)
        time.sleep(wait_seconds)
    raise requests.HTTPError(
        f"still rate limited after {max_rate_limited_waits} waits: {url}",
        response=response,
    )


def _retry_after_seconds(response: requests.Response) -> float:
    """Seconds to wait after a 429: its Retry-After header in seconds
    (at most _MAX_RATE_LIMITED_WAIT_SECONDS), else
    _DEFAULT_RATE_LIMITED_WAIT_SECONDS.

    Inputs: response (a 429). Output: seconds. Side effects: none.
    Exceptions: none.
    """
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after.isdigit():
        return min(float(retry_after), _MAX_RATE_LIMITED_WAIT_SECONDS)
    return _DEFAULT_RATE_LIMITED_WAIT_SECONDS


def pick_keys(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The entries of source under keys, in keys' order, skipping absent
    ones: the pruning step API crawls use to unwrap a payload.

    Inputs: source, keys. Output: a new dict. Side effects: none.
    Exceptions: none.

    Example:
        >>> pick_keys({"a": 1, "b": 2}, ("b", "c"))
        {'b': 2}
    """
    return {key: source[key] for key in keys if key in source}


def require_int(value: Any) -> int:
    """value if it is an int (not a bool): the check for an int read back
    from JSON a downloader wrote (a cursor, a plan).

    Inputs: value (a parsed JSON value). Output: int. Side effects: none.
    Exceptions: TypeError otherwise.

    Example:
        >>> require_int(3)
        3
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an int, got {value!r}")
    return value


def require_str(value: Any) -> str:
    """value if it is a str: the check for a string read back from JSON a
    downloader wrote (a plan's deck paths).

    Inputs: value (a parsed JSON value). Output: str. Side effects: none.
    Exceptions: TypeError otherwise.

    Example:
        >>> require_str("325-Burn")
        '325-Burn'
    """
    if not isinstance(value, str):
        raise TypeError(f"expected a str, got {value!r}")
    return value


def check_sha256(path: Path, expected_sha256: str) -> None:
    """Delete path and raise if its SHA-256 isn't expected_sha256 (for a
    source that publishes a digest per file: MTGJSON, tcgjson).

    Inputs: path, expected_sha256 (lowercase hex). Output: none.
    Side effects: reads path; deletes it on a mismatch.
    Exceptions: ValueError on a mismatch; OSError if path can't be read.

    Example:
        >>> check_sha256(Path("data/raw/tcgjson/r/lorcana.full.json.gz"), "b54d...")
    """
    actual_sha256 = sha256_of(path)
    if actual_sha256 != expected_sha256:
        path.unlink()
        raise ValueError(
            f"{path.name}: SHA-256 {actual_sha256}, expected {expected_sha256}; deleted"
        )


def sha256_of(path: Path) -> str:
    """path's lowercase hex SHA-256, read in chunks.

    Inputs: path. Output: str. Side effects: reads path.
    Exceptions: OSError if path can't be read.

    Example:
        >>> sha256_of(Path("data/raw/tcgjson/r/lorcana.full.json.gz"))
        'b54d0a...'
    """
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
