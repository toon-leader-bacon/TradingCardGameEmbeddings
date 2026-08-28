"""Streams an HTTP GET response to a file on disk.

Shared across data_retrieval sources — extracted after the same
stream-to-disk-with-cleanup block showed up independently in three
downloaders (Scryfall, Pokemon TCG, HearthstoneJSON), each with its
own copy of the chunk-size/timeout constants. Placed at this shared
level for the same reason as rate_limiter.py: no source-specific
coupling, and duplicated magic numbers across three copies risk
silently drifting apart. See src/data_retrieval/README.md.
"""

from pathlib import Path

import requests

_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_CHUNK_SIZE_BYTES = (
    1024 * 1024
)  # 1 MiB, so large responses never sit fully in memory


def download_to_file(
    url: str,
    destination_path: Path,
    *,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    chunk_size_bytes: int = _DEFAULT_CHUNK_SIZE_BYTES,
) -> None:
    """Stream a GET response to destination_path, in chunks.

    Inputs:
        url: URL to GET.
        destination_path: file path the response body is written to.
        timeout_seconds: request timeout, in seconds.
        chunk_size_bytes: size of each streamed write, in bytes.
    Output: none.
    Side effects: one network request; creates destination_path's
        parent directory if missing; writes destination_path.
    Exceptions: raises on network failure (e.g. connection error,
        non-2xx response) or on failure to write the file. Any
        partially-written file is removed before the exception
        propagates.

    Example:
        >>> download_to_file(
        ...     "https://example.com/file.zip",
        ...     Path("data/raw/example/file.zip"),
        ... )
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            with open(destination_path, "wb") as destination_file:
                for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                    destination_file.write(chunk)
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise
