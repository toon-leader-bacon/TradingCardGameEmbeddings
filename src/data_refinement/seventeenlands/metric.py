"""The shared contract every 17lands metric satisfies, driven by
CsvScanner (csv_scanner.py). See README.md's "Accumulation metric
builders" vs "Streaming metric builders" for what accumulate()/
finalize() mean for each kind.
"""

from pathlib import Path
from typing import Protocol

import pandas as pd


class Metric(Protocol):
    """A single metric driven over one raw CSV, chunk by chunk.

    Structural, not a base class — any object with this shape (e.g.
    AveragePickNumberMetric, PickedVHeldVPackMetric) satisfies it
    without inheriting from it.
    """

    @property
    def name(self) -> str: ...

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """Fold one chunk of raw data into this metric's own state."""
        ...

    def finalize(self) -> Path:
        """Write this metric's output to its own file and return that path."""
        ...


def metric_output_path(
    default_dir: Path,
    default_name: str,
    expansion: str,
    format_code: str,
    output_dir: Path | None = None,
    output_name: str | None = None,
) -> Path:
    """Build a metric's output path from expansion/format_code, falling
    back to a metric's own class-level defaults when output_dir/
    output_name aren't overridden. Shared by every Metric's own
    output_path() static method, to avoid re-deriving this per metric.
    """
    if output_dir is None:
        output_dir = default_dir
    if output_name is None:
        output_name = default_name
    return output_dir / output_name.format(expansion=expansion, format_code=format_code)
