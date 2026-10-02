"""Parity check: compares candidate metric outputs against reference
outputs, file by file (the parity rule for porting a metric; see
src/data_refinement/metrics/seventeenlands/game_data/README.md).

Typical use, for one 17lands CSV:

    # 1. New code writes to a scratch root (never over the references)
    PYTHONPATH=. python3 scripts/run_metrics.py --source seventeenlands_game_data \\
        --raw-path data/raw/17lands/game_data/KTK.TradDraft.csv \\
        --output-root scratch/parity
    # 2. Diff every output the candidate tree holds
    PYTHONPATH=. python3 scripts/compare_metric_outputs.py \\
        --reference data/metrics/seventeenlands/game_data/KTK/TradDraft \\
        --candidate scratch/parity/seventeenlands/game_data/KTK/TradDraft

Every parquet under either root is checked; a file under only one root
is MISSING CANDIDATE (the port wrote nothing - e.g. its finalize()
failed and was only logged) or MISSING REFERENCE.

A pair matches when:
- both carry the same stamped card_binder_version. A differing stamp is
  reported as STALE REFERENCE, not a mismatch: regenerate the reference
  by running the pre-port commit with --output-root, then compare again;
- they have the same columns;
- after both are sorted by their key columns, the rows line up. Key
  columns are every string column (nocab_uuid; deck ids in later
  slices);
- every integer and boolean column (e.g. sample_count) is equal;
- every float column is within --tolerance (default 1e-9 absolute).

A reference with no columns and no rows (the row implementation's
empty-CSV output) matches a zero-row candidate with any schema.

Exit status: 0 if every pair matches, 1 otherwise. Prints one line per
file.
"""

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.version_metadata import read_version_metadata

_DEFAULT_TOLERANCE = 1e-9


class ParityStatus(Enum):
    """The outcome of comparing one reference/candidate file pair."""

    MATCH = "match"
    MISMATCH = "mismatch"
    STALE_REFERENCE = "stale reference"
    MISSING_REFERENCE = "missing reference"
    MISSING_CANDIDATE = "missing candidate"


@dataclass(frozen=True)
class ParityResult:
    """One file pair's comparison.

    relative_path: the file's path under both roots.
    status: the outcome.
    detail: a short human-readable reason (empty on MATCH).
    """

    relative_path: Path
    status: ParityStatus
    detail: str


def compare_trees(
    reference_root: Path, candidate_root: Path, tolerance: float
) -> list[ParityResult]:
    """Compare every parquet under either root with its twin at the same
    relative path under the other.

    Inputs: reference_root, candidate_root (directories), tolerance
        (absolute, for float columns).
    Output: one ParityResult per relative path found under either root,
        sorted by path.
    Side effects: reads parquet files.
    Exceptions: FileNotFoundError if either root is missing.

    Example:
        >>> compare_trees(Path("data/metrics/.../KTK/TradDraft"),
        ...               Path("scratch/parity/.../KTK/TradDraft"), 1e-9)
    """
    result: list[ParityResult] = []

    # Validate inputs
    for root in (reference_root, candidate_root):
        if not root.is_dir():
            raise FileNotFoundError(root)

    # Every relative path under either root, each compared once
    for relative_path in _relative_parquet_paths(reference_root, candidate_root):
        reference_path = reference_root / relative_path
        candidate_path = candidate_root / relative_path
        if not reference_path.exists():
            result.append(
                ParityResult(relative_path, ParityStatus.MISSING_REFERENCE, "")
            )
            continue
        if not candidate_path.exists():
            result.append(
                ParityResult(relative_path, ParityStatus.MISSING_CANDIDATE, "")
            )
            continue
        result.append(
            compare_files(reference_path, candidate_path, relative_path, tolerance)
        )

    return result


def _relative_parquet_paths(reference_root: Path, candidate_root: Path) -> list[Path]:
    """The sorted union of every *.parquet path, relative to its root,
    under both roots.

    Inputs: both roots. Output: sorted list of relative Paths.
    Side effects: walks both directories. Exceptions: none.
    """
    paths = {
        path.relative_to(root)
        for root in (reference_root, candidate_root)
        for path in root.rglob("*.parquet")
    }
    return sorted(paths)


def compare_files(
    reference_path: Path, candidate_path: Path, relative_path: Path, tolerance: float
) -> ParityResult:
    """Compare one reference/candidate parquet pair under the parity
    rule (see module docstring).

    Inputs: reference_path, candidate_path, relative_path (for the
        result), tolerance.
    Output: ParityResult.
    Side effects: reads both files.
    Exceptions: whatever reading a parquet raises.

    Example:
        >>> compare_files(ref, cand, Path("win_rate_when_in_deck.parquet"), 1e-9)
    """
    # A stamp mismatch means the reference is stale, not the port wrong
    if not _same_binder_version(reference_path, candidate_path):
        return ParityResult(
            relative_path, ParityStatus.STALE_REFERENCE, "binder version"
        )

    reference = pd.read_parquet(reference_path)
    candidate = pd.read_parquet(candidate_path)

    # The row implementation's empty output had no columns at all
    if _is_empty_equivalent(reference, candidate):
        return ParityResult(relative_path, ParityStatus.MATCH, "")

    # Align both on their key columns, then check each column's values
    mismatch = _first_mismatch(reference, candidate, tolerance)
    if mismatch is not None:
        return ParityResult(relative_path, ParityStatus.MISMATCH, mismatch)
    return ParityResult(relative_path, ParityStatus.MATCH, "")


def _same_binder_version(reference_path: Path, candidate_path: Path) -> bool:
    """Whether both files carry the same stamped card_binder_version
    (read_version_metadata from metrics/version_metadata.py).

    Inputs: both paths. Output: bool (False if either has no stamp).
    Side effects: reads both files' schemas. Exceptions: read errors.
    """
    reference = read_version_metadata(reference_path)
    candidate = read_version_metadata(candidate_path)
    if reference is None or candidate is None:
        return False
    return reference.card_binder_version == candidate.card_binder_version


def _is_empty_equivalent(reference: pd.DataFrame, candidate: pd.DataFrame) -> bool:
    """True for a column-less, row-less reference beside a zero-row
    candidate.

    Inputs: both frames. Output: bool.
    Side effects: none. Exceptions: none.
    """
    return reference.shape == (0, 0) and len(candidate) == 0


def _first_mismatch(
    reference: pd.DataFrame, candidate: pd.DataFrame, tolerance: float
) -> str | None:
    """The first parity violation between two non-empty-equivalent
    frames: different columns or row counts, rows that don't line up
    once both are sorted by their key (string) columns, an unequal
    integer or boolean cell, or a float cell further apart than
    tolerance.

    Inputs: both frames, tolerance. Output: a short reason, or None
        when they match.
    Side effects: none. Exceptions: none.
    """
    # Same columns and row count before anything else
    if sorted(reference.columns) != sorted(candidate.columns):
        return f"columns {sorted(reference.columns)} vs {sorted(candidate.columns)}"
    if len(reference) != len(candidate):
        return f"{len(reference)} rows vs {len(candidate)}"

    # Line both up on their key (string) columns
    keys = [c for c in reference.columns if _is_key_column(reference[c])]
    reference = reference.sort_values(keys).reset_index(drop=True)
    candidate = candidate.sort_values(keys).reset_index(drop=True)[
        list(reference.columns)
    ]

    # Each column: exact for keys/ints/bools, within tolerance for floats
    for column in reference.columns:
        expected, actual = reference[column], candidate[column]
        if pd.api.types.is_float_dtype(expected):
            gap = (expected - actual).abs()
            both_nan = expected.isna() & actual.isna()
            worst = float(gap[~both_nan].max()) if (~both_nan).any() else 0.0
            if gap[~both_nan].isna().any() or worst > tolerance:
                return f"{column}: max |diff| {worst:.3g} > {tolerance:g}"
        elif not expected.equals(actual):
            differing = int((expected != actual).sum())
            return f"{column}: {differing} rows differ"
    return None


def _is_key_column(values: pd.Series) -> bool:
    """Whether values is a key column: string (object) dtype.

    Inputs: values. Output: bool. Side effects: none. Exceptions: none.
    """
    return pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Inputs: sys.argv. Output: Namespace with reference, candidate,
        tolerance.
    Side effects: argparse may print usage and exit. Exceptions: none.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    """Compare the two trees, print one line per file, exit 1 unless
    every pair matches.

    Inputs: command line. Output: none.
    Side effects: prints; exits. Exceptions: see compare_trees.
    """
    args = parse_args()
    results = compare_trees(args.reference, args.candidate, args.tolerance)

    for parity in results:
        print(f"{parity.status.value:18} {parity.relative_path} {parity.detail}")

    all_match = bool(results) and all(r.status is ParityStatus.MATCH for r in results)
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
