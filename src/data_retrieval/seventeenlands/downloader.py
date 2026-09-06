"""Downloads and extracts 17Lands' public per-set/per-format CSV dumps.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering (that's
data_refinement's job).

Best-effort, collect-and-continue: one ref failing (network error
during download, corrupt gzip during extraction) does not abort the
rest of a batch — every ref's outcome is recorded in
DownloadBatchResult, same convention as HearthstoneJsonDownloader's
BuildDownloadOutcome. See refs.py's module docstring for how refs are
produced (LandingPageParser, SeventeenLandsFileRef.from_known(), or
known_files.list_known_refs()) — download() falls back to the last of
those when given no refs (or an empty refs list).
"""

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tqdm import tqdm

from src.data_retrieval.download_utils import download_to_file
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.seventeenlands.known_files import list_known_refs
from src.data_retrieval.seventeenlands.refs import DataType, SeventeenLandsFileRef


@dataclass(frozen=True)
class DownloadOutcome:
    """The result of downloading one SeventeenLandsFileRef.

    Exactly one of path/error is meaningful per outcome (path set on
    success, error set on failure) — see SeventeenLandsDownloader
    docstrings for which. Not expressed as a discriminated union here;
    flagged to the human as a possible future tightening, consistent
    with BuildDownloadOutcome's Success/Failure union in
    hearthstonejson/downloader.py.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    ref: SeventeenLandsFileRef
    path: Path | None  # None if this ref failed
    error: Exception | None  # None if this ref succeeded


@dataclass(frozen=True)
class DownloadBatchResult:
    """Every outcome from one SeventeenLandsDownloader.download() call.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    outcomes: list[DownloadOutcome]


class SeventeenLandsDownloader:
    """Downloads and extracts a filtered batch of 17Lands data files.

    Single-consumer to src/data_retrieval/seventeenlands/ — no other
    source directory depends on this class.
    """

    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/17lands")

    def __init__(
        self, raw_data_dir: Path | None = None, *, rate_limiter: RateLimiter
    ) -> None:
        """
        Inputs:
            raw_data_dir: directory downloaded/extracted files are
                written under. Defaults to DEFAULT_RAW_DATA_DIR when
                omitted (expected to be data/raw/17lands, per this
                container's README — not this class's concern to
                enforce, just to receive). Each file lands at
                raw_data_dir/<data_type>/<expansion>.<format_code>.csv.
            rate_limiter: paces every outgoing download request this
                class makes. Passed in rather than constructed
                internally (dependency injection — PATTERNS.md), same
                convention as HearthstoneJsonDownloader.
        Output: none (constructor).
        Side effects: none — no I/O happens until download() or
            download_one() is called.
        Exceptions: none.
        """
        self.raw_data_dir = (
            raw_data_dir if raw_data_dir is not None else self.DEFAULT_RAW_DATA_DIR
        )
        self.rate_limiter = rate_limiter

    def download(
        self,
        refs: list[SeventeenLandsFileRef] | None = None,
        *,
        data_types: list[DataType] | None = None,
        expansions: list[str] | None = None,
        formats: list[str] | None = None,
    ) -> DownloadBatchResult:
        """Filter refs, then download+extract each surviving one.

        refs is optional — omitting it (or passing an empty list) means
        "every known-valid file" rather than "nothing" — it's expanded
        to known_files.list_known_refs()
        before filtering, so a caller with no specific refs in hand
        can still pass expansions=[...]/formats=[...] to narrow that
        full set down. Composed of _filter_refs() followed by one
        download_one() call per surviving ref — no additional logic
        beyond calling the two and collecting results. Best-effort: one
        ref's failure is recorded and the loop continues to the next
        ref, rather than aborting the whole batch — see download_one().

        Inputs:
            refs: candidate files to download. None (the default) or
                an empty list both mean every ref in
                known_files.list_known_refs().
            data_types: if given, only refs whose data_type is in this
                list are downloaded. None means no filtering on this
                dimension (all data types included).
            expansions: if given, only refs whose expansion is in this
                list are downloaded. None means no filtering on this
                dimension.
            formats: if given, only refs whose format_code is in this list
                are downloaded. None means no filtering on this
                dimension.
        Output: one DownloadOutcome per ref that survived filtering, in
            the same order as the filtered refs.
        Side effects: creates raw_data_dir (and per-data-type
            subdirectories) if missing; one paced network request and
            (on success) two file writes (compressed download, then
            extracted CSV) per surviving ref; prints a tqdm progress
            bar to stderr covering the filtered refs, and one
            tqdm.write() line per failed ref — same convention as
            HearthstoneJsonDownloader.download_missing_builds().
        Exceptions: none — download_one() failures are caught here, not
            propagated (see download_one()).

        Example:
            >>> downloader = SeventeenLandsDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> refs = [SeventeenLandsFileRef.from_known(DataType.GAME, "MSH", "PremierDraft")]
            >>> result = downloader.download(refs)
            >>> everything = downloader.download()  # every known-valid file
        """
        candidate_refs = refs if refs else list_known_refs()
        filtered_refs = self._filter_refs(
            candidate_refs,
            data_types=data_types,
            expansions=expansions,
            formats=formats,
        )

        outcomes: list[DownloadOutcome] = []
        for ref in tqdm(filtered_refs, desc="17Lands files", unit="file"):
            try:
                path = self.download_one(ref)
            except Exception as error:
                tqdm.write(f"17Lands file {ref.url} failed: {error}")
                outcomes.append(DownloadOutcome(ref=ref, path=None, error=error))
            else:
                outcomes.append(DownloadOutcome(ref=ref, path=path, error=None))

        return DownloadBatchResult(outcomes=outcomes)

    def _filter_refs(
        self,
        refs: list[SeventeenLandsFileRef],
        *,
        data_types: list[DataType] | None,
        expansions: list[str] | None,
        formats: list[str] | None,
    ) -> list[SeventeenLandsFileRef]:
        """Keep only refs matching every given (non-None) filter dimension.

        Private helper — single consumer is download(). A None filter
        dimension excludes that dimension from consideration entirely
        (does not mean "match nothing").

        Inputs:
            refs: candidate refs.
            data_types: allowed data types, or None for no filtering.
            expansions: allowed expansion codes, or None for no
                filtering.
            formats: allowed format_code values, or None for no filtering.
        Output: refs whose data_type/expansion/format_code each satisfy
            their corresponding filter (or every ref, if all three
            filters are None), preserving input order.
        Side effects: none.
        Exceptions: none.
        """
        return [
            ref
            for ref in refs
            if (data_types is None or ref.data_type in data_types)
            and (expansions is None or ref.expansion in expansions)
            and (formats is None or ref.format_code in formats)
        ]

    def download_one(self, ref: SeventeenLandsFileRef) -> Path:
        """Download and extract one 17Lands file.

        Downloads ref.url to a temporary .csv.gz (via the shared
        download_to_file, paced by self.rate_limiter), then extracts it
        to raw_data_dir/<data_type>/<expansion>.<format_code>.csv — same
        download-then-extract shape as
        ScryfallOracleDownloader.download()/.extract(), collapsed into
        one method here since (unlike Scryfall) callers of this class
        only ever want the extracted CSV, never the intermediate
        .csv.gz.

        Inputs:
            ref: which file to download.
        Output: path to the extracted, decompressed CSV file.
        Side effects: one paced network request; creates
            raw_data_dir/<ref.data_type>/ if missing; writes the
            compressed file, then the extracted file, then removes the
            compressed file. Any partially-written file (compressed or
            extracted) is removed before an exception propagates, same
            convention as download_to_file/ScryfallOracleDownloader.
        Exceptions: raises on network failure (e.g. connection error,
            non-2xx response), on failure to write either file, or if
            the downloaded content isn't valid gzip.

        Example:
            >>> downloader = SeventeenLandsDownloader(
            ...     rate_limiter=RateLimiter(requests_per_minute=12),
            ... )
            >>> ref = SeventeenLandsFileRef.from_known(DataType.GAME, "MSH", "PremierDraft")
            >>> downloader.download_one(ref)
            PosixPath('data/raw/17lands/game_data/MSH.PremierDraft.csv')
        """
        destination_path = self._destination_path(ref)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        compressed_path = destination_path.with_suffix(".csv.gz")

        self.rate_limiter.wait()
        download_to_file(ref.url, compressed_path)

        try:
            with gzip.open(compressed_path, "rb") as compressed_file:
                with open(destination_path, "wb") as destination_file:
                    shutil.copyfileobj(compressed_file, destination_file)
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise
        finally:
            compressed_path.unlink(missing_ok=True)

        return destination_path

    def _destination_path(self, ref: SeventeenLandsFileRef) -> Path:
        """Compute the extracted CSV's on-disk path for a given ref.

        Private helper — single consumer is download_one(). Mirrors
        the S3 key structure: raw_data_dir/<data_type>/<expansion>.
        <format_code>.csv.

        Inputs:
            ref: which file's destination path to compute.
        Output: path under raw_data_dir the extracted CSV belongs at.
        Side effects: none — does not create any directory or check
            existence.
        Exceptions: none.
        """
        return (
            self.raw_data_dir
            / ref.data_type.value
            / f"{ref.expansion}.{ref.format_code}.csv"
        )
