"""Shared results-writer for every *MetricScanner — the row-building,
dtype-cast, and parquet-write sequence every MetricScanner/
DraftMetricScanner/ReplayMetricScanner delegates to instead of each
implementing its own.

Deliberately does NOT construct or return a *MetricScanResult itself —
each pipeline's ScanResult dataclass carries a legitimately
different-shaped "unresolved" field (list[str] for game/draft,
frozenset[str] for replay — see replay_data_metrics/README.md for why
that one deviates), so unifying that shape would force a
coupled abstraction over something that's only superficially the same
(PRINCIPLES.md section 2). Each Scanner still constructs its own
ScanResult after calling this function, threading through whatever
unresolved value it already computed.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data_refinement.seventeenlands.scannable_metric import ScannableMetric


def write_metric_results(metrics: Sequence[ScannableMetric], output_path: Path) -> None:
    """Finalize every metric and write all MetricResult rows to output_path.

    Calls finalize() on every metric in metrics and merges every
    metric's MetricResult rows into one parquet file — one row per
    (nocab_uuid, metric_name) pair. Columns are cast explicitly
    (nocab_uuid/metric_name/expansion/format as "string", value as
    "float64", sample_size as "int64") even when zero rows are
    produced — an empty result set otherwise leaves pandas nothing to
    infer dtypes from, defaulting every column to object, which would
    make an empty scan's parquet schema silently disagree with a
    non-empty scan's.

    Inputs:
        metrics: every metric to finalize and write. Order doesn't
            affect the output (rows are keyed by (nocab_uuid,
            metric_name), not position).
        output_path: where to write the merged parquet file.
    Output: none.
    Side effects: creates output_path's parent directory if missing;
        writes/overwrites output_path.
    Exceptions: raises on failure to write output_path, or whatever a
        metric's finalize() raises.

    Example:
        >>> write_metric_results([win_rate_metric, drawn_win_rate_metric], Path("out.parquet"))
    """
    rows = [
        {
            "nocab_uuid": str(result.nocab_uuid),
            "metric_name": result.metric_name,
            "value": result.value,
            "sample_size": result.sample_size,
            "expansion": result.expansion,
            "format": result.format,
        }
        for metric in metrics
        for result in metric.finalize().values()
    ]
    results_df = pd.DataFrame(
        rows,
        columns=[
            "nocab_uuid",
            "metric_name",
            "value",
            "sample_size",
            "expansion",
            "format",
        ],
    )
    # An empty `rows` (e.g. no active metrics, or every metric produced
    # zero results) leaves pandas nothing to infer dtypes from, so
    # every column defaults to object — an explicit cast keeps the
    # written parquet schema identical whether or not this scan
    # happened to produce any rows.
    results_df = results_df.astype(
        {
            "nocab_uuid": "string",
            "metric_name": "string",
            "value": "float64",
            "sample_size": "int64",
            "expansion": "string",
            "format": "string",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)
