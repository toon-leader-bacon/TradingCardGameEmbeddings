"""Shared row-counting utility for CSV-streaming *MetricScanners.

MetricScanner and ReplayMetricScanner already duplicate this exact logic
identically today (replay_metric_scanner.py's prior docstring explicitly
named this as a rule-of-three candidate). DraftMetricScanner does NOT
call anything like this today — see tmp/REFACTOR.md §2's "Unexplained
sibling inconsistency," still open. Adopting metric_scan_loop.py (which
calls this function) is what makes DraftMetricScanner the third real
consumer and brings its progress bar up to match its two siblings — this
module's extraction is a target-state move, not a description of
DraftMetricScanner's current behavior. Used to give tqdm a total (and
thus a percentage/ETA) instead of a bare chunk counter — see
metric_scan_loop.py, the one consumer of this function.
"""

from pathlib import Path


def count_data_rows(csv_path: Path) -> int:
    """Estimate csv_path's data row count via a raw newline scan.

    Counts newline bytes directly rather than parsing the CSV, so it
    stays cheap even on a multi-GB file; a field containing an
    embedded newline would make this an estimate rather than an exact
    count, which is acceptable for a progress display (never used for
    correctness-critical logic — checkpoint resume relies on the
    scanner's own cumulative rows_processed count, not on this).

    Inputs:
        csv_path: path to the CSV file to count.
    Output: number of data rows, i.e. newline count minus the header
        row (never negative).
    Side effects: reads csv_path in full, as raw bytes (no parsing).
    Exceptions: raises if csv_path doesn't exist or can't be read.

    Example:
        >>> count_data_rows(Path("data/raw/17lands/game_data/MSH.PremierDraft.csv"))
    """
    with csv_path.open("rb") as f:
        newline_count = sum(
            buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b"")
        )
    return max(newline_count - 1, 0)
