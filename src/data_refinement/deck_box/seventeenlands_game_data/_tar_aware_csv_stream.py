"""Tar-aware streaming I/O for game_data CSVs, some of which are tar-wrapped on disk.

Single consumer: src/data_refinement/deck_box/seventeenlands_game_data/extraction_stage.py.
See that module's docstring (TAR-WRAPPED FILES and STREAMING sections)
for why this detection and dual-mode opening exists.
"""

import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

# POSIX ustar header layout: the "ustar" magic sits at byte offset 257
# of every 512-byte tar header - same constant SeventeenLandsDownloader
# now checks at download time (see extraction_stage.py's module
# docstring's TAR-WRAPPED FILES section for why this is re-checked here
# rather than assumed fixed).
_TAR_MAGIC_OFFSET = 257
_TAR_MAGIC = b"ustar"


def _is_tar_wrapped(csv_path: Path) -> bool:
    """Detect whether csv_path is actually a tar archive on disk.

    Private helper — single consumer is _open_stream(). Peeks the
    first 512 bytes and checks for the ustar magic at its known
    offset — same technique and constants
    SeventeenLandsDownloader.download_one() now uses at download
    time (see extraction_stage.py's module docstring's TAR-WRAPPED
    FILES section).

    Inputs:
        csv_path: path to check.
    Output: True if csv_path's first 512 bytes carry a ustar tar
        header; False otherwise (a plain CSV).
    Side effects: none — reads only the first 512 bytes.
    Exceptions: none expected.
    """
    with open(csv_path, "rb") as probe_file:
        header = probe_file.read(512)
    return header[_TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + len(_TAR_MAGIC)] == (
        _TAR_MAGIC
    )


@contextmanager
def _open_stream(csv_path: Path) -> Iterator[tuple[IO[bytes], int]]:
    """Open csv_path for repeated binary reads, tar-wrapped or not.

    Private helper — single consumer is extraction_stage.py's
    _extract_csv(). Dispatches on _is_tar_wrapped(): for a tar-wrapped
    file, opens it via tarfile.open(csv_path, mode="r") (random-access
    mode — the on-disk quirk here is already-decompressed, so this is
    a plain seekable file, unlike download_one()'s in-flight gzip
    stream) and yields its one member's extractfile() stream (which
    itself supports .seek()/.tell() since the underlying tar file
    does); for a plain file, yields a standard open(csv_path, "rb").
    Both branches yield a stream _extract_csv() reads TWICE — once via
    pandas for the header (nrows=0), then seek(0) back to the start
    for the real chunked pass — so the returned stream must support
    seek(0) either way.

    Inputs:
        csv_path: the file to open.
    Output (yielded): (stream, total_bytes) — stream is the binary
        handle to read; total_bytes is the CSV content's own byte
        length (the tar member's declared size when tar-wrapped, not
        csv_path's on-disk size, which for a tar-wrapped file
        includes header/padding overhead).
    Side effects: opens csv_path (and, when tar-wrapped, the TarFile
        it belongs to); both are closed on context exit.
    Exceptions: raises ValueError if csv_path is tar-wrapped but its
        tar has no members, or its first member isn't a regular
        file (mirrors
        SeventeenLandsDownloader._extract_tar_member()'s own guards).

    Example:
        >>> with _open_stream(csv_path) as (stream, total_bytes):
        ...     header = pd.read_csv(stream, nrows=0).columns
        ...     stream.seek(0)
    """
    if _is_tar_wrapped(csv_path):
        with tarfile.open(csv_path, mode="r") as tar:
            member = tar.next()
            if member is None:
                raise ValueError(
                    f"_open_stream: {csv_path} is tar-wrapped but has no members"
                )
            member_file = tar.extractfile(member)
            if member_file is None:
                raise ValueError(
                    f"_open_stream: {csv_path}'s first member "
                    f"{member.name!r} isn't a regular file"
                )
            yield member_file, member.size
    else:
        with open(csv_path, "rb") as csv_file:
            yield csv_file, csv_path.stat().st_size
